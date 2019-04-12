module Transforms

using StaticArrays

export Transform
export Empty, Shift


"""
    Transform{M, N, R}

Represents a transformation from M-dimensional to R-dimensional space
with elements of type R.
"""
abstract type Transform{From, To, R<:Real} end

@inline fromdims(::Type{<:Transform{F}}) where {F} = F
@inline todims(::Type{<:Transform{_F, T}}) where {_F, T} = T
@inline eltype(::Type{<:Transform{_F, _T, R}}) where {_F, _T, R} = R

@inline fromdims(t::T) where T<:Transform = fromdims(T)
@inline todims(t::T) where T<:Transform = todims(T)
@inline eltype(t::T) where T<:Transform = eltype(T)


"""
    Empty{D,R}()

A D-dimensional transform that does nothing.
"""
struct Empty{D, R} <: Transform{D, D, R} end

# Convenient but type-unstable constructors
Empty(D, R) = Empty{D, R}()
Empty(D) = Empty(D, Float64)

@inline (self::Empty)(point) = point
@inline (self::Empty)(point, grad) = (point, grad)


"""
    Shift(x::SVector)

Create a shifting transformation that adds *x* to each input vector.
"""
struct Shift{D, R} <: Transform{D, D, R}
    offset :: SVector{D, R}
    Shift(offset::SVector{D,R}) where {D,R} = new{D,R}(offset)
end

@inline (self::Shift)(point) = point + self.offset
@inline (self::Shift)(point, grad) = (point + self.offset, grad)


end # module
