module Utils

export outer, exterior, MemoizedMap


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

function Base.getindex(self::MemoizedMap, i::Int)
    i in keys(self.map) && return self.map[i]
    self.map[i] = self.next
    let retval = self.next
        self.next += 1
        retval
    end
end

(self::MemoizedMap)(i::Int) = self[i]


function outer(factors...)
    .*((reshape(f, fill(1, k-1)..., :) for (k, f) in enumerate(factors))...)
end

exterior(func; repeat=2) = outer(Iterators.repeated(func, repeat)...)

end
