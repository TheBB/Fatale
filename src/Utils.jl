module Utils

import Base: getindex

export outer, exterior, MemoizedMap, swap, newaxis


const newaxis = [CartesianIndex()]


"""
    MemoizedMap([mapping])

A mapping that generates integers, initialized with the contents of
*mapping*, if given. When requesting a key that does not exist, will
generate a new unique one and remember it.

Supports both getindex and function call syntax.
"""
mutable struct MemoizedMap{T}
    map :: T
    next :: Int

    function MemoizedMap(map::T) where T
        @assert keytype(map) == Int
        @assert valtype(map) == Int
        new{T}(map, maximum(values(map)) + 1)
    end
end

function getindex(self::MemoizedMap, i::Int)
    i in keys(self.map) && return self.map[i]
    self.map[i] = self.next
    let retval = self.next
        self.next += 1
        retval
    end
end

(self::MemoizedMap)(i::Int) = self[i]


"""
    swap!(collection, i, j)

Swap two elements in a collection. Same as

     collection[i], collection[j] = collection[j], collection[i]
"""
function swap!(vec, i, j)
    vec[i], vec[j] = vec[j], vec[i]
end


"""
    outer(factors...)

Outer product of a number of vector-valued factors.
"""
function outer(factors...)
    .*((reshape(f, fill(1, k-1)..., :) for (k, f) in enumerate(factors))...)
end


"""
    exterior(left[, right])

Forms an outer product of the last axes of each argument. If only one
argument is given, form the outer product with itself.
"""
function exterior(left, right)
    @assert ndims(left) == ndims(right)
    @assert size(left, ndims(left)) == size(right, ndims(right))
    n = size(left, ndims(left))
    left = reshape(left, size(left)[1:end-1]..., n, 1)
    right = reshape(right, size(right)[1:end-1]..., 1, n)
    left .* right
end

exterior(self) = exterior(self, self)

end
