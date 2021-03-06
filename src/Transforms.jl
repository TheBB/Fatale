module Transforms

using Base: @_inline_meta
using Base.Iterators: product
using StaticArrays: SVector, SMatrix
using LinearAlgebra: I
using ..Utils: newaxis

import Base: ∘, eltype, length, getindex

export AbstractTransform, todims, fromdims, isupdim, apply
export Empty, Affine
export shift, updim


const Coords{P,G} = NamedTuple{(:point, :grad), Tuple{P,G}}


"""
    AbstractTransform{M, N, R}

Represents a transformation from M-dimensional to N-dimensional space
with elements of type R.

Concrete subtypes should implement callable methods:

    (trf::AbstractTransform)(point)

Apply the transformation *trf* to the point *point* and return the
transformed point.

    (trf::AbstractTransform)(point, grad)

Apply the transformation *trf* to the point *point* and the derivative
matrix *grad* (typically the identity) and return the new point and
derivative.
"""
abstract type AbstractTransform{From, To, R<:Real} end

@inline fromdims(::Type{<:AbstractTransform{F}}) where F = F
@inline todims(::Type{<:AbstractTransform{_F, T}}) where {_F, T} = T
@inline eltype(::Type{<:AbstractTransform{_F, _T, R}}) where {_F, _T, R} = R
@inline isupdim(::Type{T}) where T<:AbstractTransform = fromdims(T) != todims(T)

@inline fromdims(t::T) where T<:AbstractTransform = fromdims(T)
@inline todims(t::T) where T<:AbstractTransform = todims(T)
@inline eltype(t::T) where T<:AbstractTransform = eltype(T)
@inline isupdim(t::T) where T<:AbstractTransform = isupdim(T)

# This is a workaround until we can define functions on abstract types
# Mostly just to convert a nothing gradient into the identity matrix
@inline apply(trf::AbstractTransform, x) = trf(x)
@inline function apply(trf::AbstractTransform, x::Coords{SVector{N,T}, Nothing}) where {N,T}
    trf((point=x.point, grad=SMatrix{N,N,T}(I)))
end


"""
    Chain(transforms)

Construct a single chain transformation from a sequence of
transformations to apply in order from right to left.
"""
struct Chain{K<:Tuple{Vararg{AbstractTransform}}, From, To, R} <: AbstractTransform{From, To, R}
    chain :: K

    function Chain(terms::Tuple{Vararg{AbstractTransform}})
        new{typeof(terms), fromdims(terms[end]), todims(terms[1]), eltype(terms[1])}(terms)
    end
end

length(self::Chain) = length(self.chain)
length(::Type{<:Chain{K}}) where K = length(K.parameters)
@inline getindex(self::Chain, i::Int) = self.chain[i]

@generated function splittrf(self::Chain{K}) where K
    i = 1
    while i <= length(self) && !isupdim(K.parameters[i])
        i += 1
    end
    if i > length(self)
        return :((self, Empty(self)))
    elseif i == 1
        return :((Empty(self), self))
    else
        return :((Chain(self.chain[1:$(i-1)]), Chain(self.chain[$i:end])))
    end
end

@generated function (self::Chain)(x)
    codes = [:(x = self[$i](x)) for i in length(self):-1:1]
    quote
        @_inline_meta
        $(codes...)
        x
    end
end


"""
    Empty{D,R}()

A D-dimensional transform that does nothing.
"""
struct Empty{D, R} <: AbstractTransform{D, D, R} end
Empty(trf::T) where T<:AbstractTransform = Empty{todims(T), eltype(T)}()

# Convenient but type-unstable constructors
Empty(D, R) = Empty{D, R}()
Empty(D) = Empty(D, Float64)

splittrf(self::Empty) = (self, self)

@inline (::Empty)(x) = x


"""
    Affine(a::SMatrix, b::SVector, flipped::Bool)

An affine transformation acting as x -> a*x + b.
"""
struct Affine{F, T, R, L} <: AbstractTransform{F, T, R}
    matrix :: SMatrix{T, F, R, L}
    vector :: SVector{T, R}
    flipped :: Bool

    function Affine(matrix::SMatrix{T, F, R}, vector::SVector{T, R}, flipped::Bool=false) where {T, F, R}
        new{F, T, R, F*T}(matrix, vector, flipped)
    end
end

@generated function splittrf(self::Affine)
    if isupdim(self)
        return :((Empty(self), self))
    else
        return :((self, Empty(self)))
    end
end

@inline (self::Affine)(point) = self.matrix * point + self.vector
@inline (self::Affine)(x::Coords) = (
    point = self.matrix * x.point + self.vector,
    grad = _exterior(x.grad * self.matrix', self.flipped)
)

@generated function _exterior(matrix, flipped)
    to = size(matrix, 2)
    from = size(matrix, 1)
    R = eltype(matrix)

    from == to && return :(matrix)
    @assert to == from + 1
    @assert 1 <= to <= 3

    # Exterior: Add a new row orthogonal to all the existing ones
    elements = [:(matrix[$i]) for i in LinearIndices(size(matrix))]

    if to == 1
        new_row = [:(one($R))]
    elseif to == 2
        (a, b) = elements
        new_row = [b, :(-$a)]
    elseif to == 3
        (a, b, c, d, e, f) = elements
        new_row = [:($c*$f - $e*$d), :($e*$b - $a*$f), :($a*$d - $c*$b)]
    end

    new_row = [:((1-2*Int(flipped))*$c) for c in new_row]
    elements = vcat(elements, new_row[newaxis, :])

    quote
        @_inline_meta
        SMatrix{$to,$to}($(elements...))
    end
end


@inline shift(x::SVector{N,T}) where {N,T} = Affine(SMatrix{N,N,T}(I), x)

@generated function updim(::Val{T}, ins::Int, value::R, flipped::Bool=false) where {T,R}
    vecelements = [:($i == ins ? value : $(zero(R))) for i in 1:T]
    mxelements = Any[
        :($(zero(R)))
        for (_, _) in product(1:T, 1:T-1)
    ]
    for i in 1:T-1
        mxelements[i,i] = :($i < ins ? $(one(R)) : $(zero(R)))
        mxelements[i+1,i] = :($i >= ins ? $(one(R)) : $(zero(R)))
    end

    quote
        @_inline_meta
        mx = SMatrix{$T,$(T-1),$R}($(mxelements...))
        vec = SVector{$T,$R}($(vecelements...))
        Affine(mx, vec, flipped)
    end
end


# Transform composition rules
@inline function compose(l::AbstractTransform, r::AbstractTransform)
    isupdim(l) && @assert isupdim(r)
    return _compose(l, r)
end
@inline compose(l::Empty, r::AbstractTransform) = r
@inline compose(l::AbstractTransform, r::Empty) = l
@inline compose(l::Empty, r::Empty) = l

@inline _compose(l::AbstractTransform, r::AbstractTransform) = Chain((l, r))
@inline _compose(l::Chain, r::AbstractTransform) = Chain((l.chain..., r))
@inline _compose(l::AbstractTransform, r::Chain) = Chain((l, r.chain...))
@inline _compose(l::Chain, r::Chain) = Chain((l.chain..., r.chain...))

# This is basically reduce(compose, trfs), except type stable
@generated function ∘(trfs::AbstractTransform...)
    l = length(trfs)
    code = :(trfs[$l])
    for i in l-1:-1:1
        code = :(compose(trfs[$i], $code))
    end

    quote
        @_inline_meta
        $code
    end
end


end # module
