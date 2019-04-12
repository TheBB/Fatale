module Transforms

import Base.Iterators: flatten
import Base: @_inline_meta
using StaticArrays

export Transform
export Empty, Shift, Updim


"""
    Transform{M, N, R}

Represents a transformation from M-dimensional to R-dimensional space
with elements of type R.
"""
abstract type Transform{From, To, R<:Real} end

@inline fromdims(::Type{<:Transform{F}}) where F = F
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


"""
    Updim{I, N}(value)

Create a transformation increasing the dimension of a space by one, by
inserting the element `value` at index `I`. The final space has
dimension `N`.
"""
struct Updim{Ins, From, To, R} <: Transform{From, To, R}
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

    src_cols = [[:(grad[$i,$j]) for i in 1:from] for j in 1:from]
    for col in src_cols
        insert!(col, Ins, :(zero($R)))
    end

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


end # module
