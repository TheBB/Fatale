import UnsafeArrays
using StaticArrays

UnsafeArrays._require_one_based_indexing(::MArray) = nothing
UnsafeArrays._require_one_based_indexing(::SArray) = nothing

Base.@propagate_inbounds UnsafeArrays.unsafe_uview(a::MArray{N,T}) where {N,T} =
    UnsafeArrays._maybe_unsafe_uview(Val{isbitstype(T)}(), a)
Base.@propagate_inbounds UnsafeArrays.unsafe_uview(a::SArray{N,T}) where {N,T} =
    UnsafeArrays._maybe_unsafe_uview(Val{isbitstype(T)}(), a)
