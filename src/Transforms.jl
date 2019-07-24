module Transforms

import Base.Iterators: flatten, product
import Base: @_inline_meta
using LinearAlgebra
using StaticArrays

using ..Utils

export AbstractTransform, todims, fromdims
export Empty, Affine
export shift, updim


const Coords{N,T,M} = NamedTuple{(:point, :grad), Tuple{SVector{N,T}, SMatrix{N,N,T,M}}}


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
@inline Base.eltype(::Type{<:AbstractTransform{_F, _T, R}}) where {_F, _T, R} = R

@inline fromdims(t::T) where T<:AbstractTransform = fromdims(T)
@inline todims(t::T) where T<:AbstractTransform = todims(T)
@inline Base.eltype(t::T) where T<:AbstractTransform = eltype(T)


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

Base.length(self::Chain) = length(self.chain)
Base.length(::Type{<:Chain{K}}) where K = length(K.parameters)
@inline Base.getindex(self::Chain, i::Int) = self.chain[i]

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

# Convenient but type-unstable constructors
Empty(D, R) = Empty{D, R}()
Empty(D) = Empty(D, Float64)

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

@inline (self::Affine)(point) = self.matrix * point + self.vector
@inline (self::Affine)(x::Coords) = (
    point = self.matrix * x.point + self.vector,
    grad = _exterior(self.matrix * x.grad, self.flipped)
)

@generated function _exterior(matrix, flipped)
    to = size(matrix, 1)
    from = size(matrix, 2)
    R = eltype(matrix)

    from == to && return :(matrix)
    @assert to == from + 1
    @assert 1 <= to <= 3

    # Exterior: Add a new column orthogonal to all the existing ones
    src_cols = [[:(matrix[$i,$j]) for i in 1:to] for j in 1:from]
    if to == 1
        new_col = [:(one($R))]
    elseif to == 2
        ((a, b),) = src_cols
        new_col = [b, :(-$a)]
    elseif to == 3
        ((a, c, e), (b, d, f)) = src_cols
        new_col = [:($c*$f - $e*$d), :($e*$b - $a*$f), :($a*$d - $c*$b)]
    end

    new_col = [:((1-2*Int(flipped))*$c) for c in new_col]

    elements = flatten((src_cols..., new_col))
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
@inline compose(l::AbstractTransform, r::AbstractTransform) = Chain((l, r))
@inline compose(l::Empty, r::AbstractTransform) = r
@inline compose(l::AbstractTransform, r::Empty) = l
@inline compose(l::Empty, r::Empty) = l

# This is basically reduce(compose, trfs), except type stable
@generated function Base.:∘(trfs::AbstractTransform...)
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
