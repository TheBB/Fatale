module Integrate

import ..Evaluables:
    OptimizedEvaluable, OptimizedBlockEvaluable, OptimizedSparseEvaluable,
    ArrayEvaluable, optimize
import ..Transforms: apply
import ..Elements: elementdata

import Strided: UnsafeStridedView, sreshape, StridedView
import SparseArrays: sparse!, dropzeros!, nnz

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


integrate(func::ArrayEvaluable, domain, quadrule) = integrate(optimize(func), domain, quadrule)


function integrate(func::OptimizedEvaluable, domain, quadrule)
    data = zeros(eltype(func), size(func))
    (pts, wts) = quadrule
    for element in domain
        loctrans = elementdata(element, Val(:loctrans))
        for (pt, wt) in zip(pts, wts)
            coords = apply(loctrans, (point=pt, grad=nothing))
            data .+= func((element=element, coords=coords, quadrule=quadrule)) .* wt
        end
    end
    data
end


function integrate(func::OptimizedSparseEvaluable{T,1}, domain, quadrule) where T
    V = zeros(T, length(func))
    for block in func.blocks
        _integrate(block, domain, quadrule, UnsafeStridedView(V))
    end
    V
end

function _integrate(block::OptimizedBlockEvaluable{1}, domain, quadrule, V)
    (pts, wts) = quadrule
    for (i, element) in enumerate(domain)
        I = block.indices[1](element, nothing)
        loctrans = elementdata(element, Val(:loctrans))
        for (pt, wt) in zip(pts, wts)
            coords = apply(loctrans, (point=pt, grad=nothing))
            V[I] .+= block.data((element=element, coords=coords, quadrule=quadrule)) .* wt
        end
    end
end


function integrate(func::OptimizedSparseEvaluable{T,2}, domain, quadrule) where T
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
        _integrate(block, domain, quadrule, It, Jt, Vt)
        i += l
    end

    _sparse!(I, J, V, size(func)...)
end

function _integrate(block::OptimizedBlockEvaluable{2}, domain, quadrule, I, J, V)
    (pts, wts) = quadrule
    for (i, element) in enumerate(domain)
        I[:,:,i] .= block.indices[1](element, nothing)
        J[:,:,i] .= block.indices[2](element, nothing)
        loctrans = elementdata(element, Val(:loctrans))
        for (pt, wt) in zip(pts, wts)
            coords = apply(loctrans, (point=pt, grad=nothing))
            V[:,:,i] .+= block.data((element=element, coords=coords, quadrule=quadrule)) .* wt
        end
    end
end


end # module
