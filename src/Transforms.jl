module Transforms

import Base.Iterators: flatten
import Base: @_inline_meta
using StaticArrays

using ..Utils

export AbstractTransform, todims, fromdims
export Empty, Shift, Updim


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

@generated function (self::Chain)(point)
    codes = [:(point = self[$i](point)) for i in length(self):-1:1]
    quote
        @_inline_meta
        $(codes...)
        point
    end
end

@generated function (self::Chain)(point, grad)
    codes = [:((point, grad) = self[$i](point, grad)) for i in length(self):-1:1]
    quote
        @_inline_meta
        $(codes...)
        (point, grad)
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

@inline (self::Empty)(point) = point
@inline (self::Empty)(point, grad) = (point, grad)


"""
    Shift(x::SVector)

A shifting transformation that adds *x* to each input vector.
"""
struct Shift{D, R} <: AbstractTransform{D, D, R}
    offset :: SVector{D, R}
    Shift(offset::SVector{D,R}) where {D,R} = new{D,R}(offset)
end

@inline (self::Shift)(point) = point + self.offset
@inline (self::Shift)(point, grad) = (point + self.offset, grad)


"""
    Updim{I, N}(value)

Create a transformation increasing the dimension of a space by one, by
inserting the element `value` at index `I`. The final space has
dimension `N`.
"""
struct Updim{Ins, From, To, R} <: AbstractTransform{From, To, R}
    value :: R

    @inline function Updim{Ins, To}(value) where {Ins, To}
        @assert 1 <= Ins <= To
        @assert 1 <= To <= 3
        new{Ins, To-1, To, typeof(value)}(value)
    end
end

@generated function (self::Updim{Ins})(point) where Ins
    elements = [:(point[$i]) for i in 1:fromdims(self)]
    insert!(elements, Ins, :(self.value))
    quote
        @_inline_meta
        @SVector [$(elements...)]
    end
end

@generated function (self::Updim{Ins})(point, grad) where Ins
    from = fromdims(self)
    to = todims(self)
    R = eltype(self)

    # Insert a new row with only zeros
    src_cols = [[:(grad[$i,$j]) for i in 1:from] for j in 1:from]
    for col in src_cols
        insert!(col, Ins, :(zero($R)))
    end

    # Exterior: Add a new column orthogonal to all the existing ones
    if to == 1
        new_col = [:(one($R))]
    elseif to == 2
        ((a, b),) = src_cols
        new_col = [b, :(-$a)]
    elseif to == 3
        ((a, c, e), (b, d, f)) = src_cols
        new_col = [:($c*$f - $e*$d), :($e*$b - $a*$f), :($a*$d - $c*$b)]
    end

    elements = flatten((src_cols..., new_col))
    quote
        @_inline_meta
        newgrad = SMatrix{$to,$to}($(elements...))
        (self(point), newgrad)
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
