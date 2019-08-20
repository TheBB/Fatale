module Integrate

using SparseArrays: sparse!, dropzeros!, nnz
using StaticArrays: SVector
using Strided: UnsafeStridedView, sreshape, StridedView

using ..Elements: elementdata
using ..Evaluables: OptimizedEvaluable, OptimizedBlockEvaluable, OptimizedSparseEvaluable
using ..Evaluables: ArrayEvaluable, optimize
using ..Transforms: Empty, apply, splittrf

export integrate, to


function _sparse!(I, J, V, m, n)
    if length(I) < n + 1
        r = length(I)+1
        resize!(I, n+1)
        resize!(J, n+1)
        resize!(V, n+1)
        I[r:end] .= one(eltype(V))
        J[r:end] .= one(eltype(V))
        V[r:end] .= zero(eltype(V))
    end

    csrrowptr = Vector{Int}(undef, m+1)
    csrcolval = Vector{Int}(undef, length(I))
    csrnzval = Vector{eltype(V)}(undef, length(I))
    klasttouch = Vector{Int}(undef, n)
    A = sparse!(I, J, V, m, n, +, klasttouch, csrrowptr, csrcolval, csrnzval, I, J, V)

    # Work around a bug that leaves more colptrs than necessary,
    # in which case dropzeros! won't work
    resize!(A.colptr, n+1)
    dropzeros!(A)
end


const QuadRule = Tuple{AbstractArray{<:SVector}, AbstractArray{Float64}}


integrate(func, domain, quadrule::QuadRule) = integrate(func, domain, (quadrule=quadrule,))
integrate(func::ArrayEvaluable, domain, args::NamedTuple) = integrate(optimize(func), domain, args)


# ==============================================================================
# Integration of dense evaluables

function integrate(func::OptimizedEvaluable, domain, args::NamedTuple)
    data = zeros(eltype(func), size(func))
    (pts, wts) = args.quadrule
    for element in domain
        loctrans = elementdata(element, Val(:loctrans))
        for (pt, wt) in zip(pts, wts)
            coords = apply(loctrans, (point=pt, grad=nothing))
            data .+= func((element=element, coords=coords, args...)) .* wt
        end
    end
    data
end


# ==============================================================================
# Integration of sparse vector evaluables

function integrate(func::OptimizedSparseEvaluable{T,1}, domain, args::NamedTuple) where T
    V = zeros(T, length(func))
    for block in func.blocks
        _integrate(block, domain, args, UnsafeStridedView(V))
    end
    V
end

function _integrate(block::OptimizedBlockEvaluable{1}, domain, args, V)
    (pts, wts) = args.quadrule
    for (i, element) in enumerate(domain)
        I = block.indices[1](element, nothing)
        _t, loctrans = splittrf(elementdata(element, Val(:loctrans)))
        @assert _t isa Empty
        for (pt, wt) in zip(pts, wts)
            coords = apply(loctrans, (point=pt, grad=nothing))
            V[I] .+= block.data((element=element, coords=coords, args...)) .* wt
        end
    end
end


# ==============================================================================
# Integration of sparse matrix evaluables

function integrate(func::OptimizedSparseEvaluable{T,2}, domain, args::NamedTuple) where T
    nelems = length(domain)
    nentries = nnz(func)

    I = Vector{Int}(undef, nentries * nelems)
    J = Vector{Int}(undef, nentries * nelems)
    V = zeros(T, nentries * nelems)

    i = 1
    for block in func.blocks
        (m,n) = size(block)
        l = length(block) * nelems
        It = sreshape(UnsafeStridedView(I)[i:l], (m, n, nelems))
        Jt = permutedims(sreshape(UnsafeStridedView(J)[i:l], (m, n, nelems)), (2, 1, 3))
        Vt = sreshape(UnsafeStridedView(V)[i:l], (m, n, nelems))
        _integrate(block, domain, args, It, Jt, Vt)
        i += l
    end

    _sparse!(I, J, V, size(func)...)
end

function _integrate(block::OptimizedBlockEvaluable{2}, domain, args, I, J, V)
    (pts, wts) = args.quadrule
    for (i, element) in enumerate(domain)
        I[:,:,i] .= block.indices[1](element, nothing)
        J[:,:,i] .= block.indices[2](element, nothing)
        _t, loctrans = splittrf(elementdata(element, Val(:loctrans)))
        @assert _t isa Empty
        for (pt, wt) in zip(pts, wts)
            coords = apply(loctrans, (point=pt, grad=nothing))
            V[:,:,i] .+= block.data((element=element, coords=coords, args...)) .* wt
        end
    end
end


end # module
