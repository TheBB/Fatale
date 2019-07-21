module Integrate

import ..Evaluables: OptimizedEvaluable, OptimizedBlockEvaluable, OptimizedSparseEvaluable

import Strided: UnsafeStridedView, sreshape
import SparseArrays: sparse!, dropzeros!, nnz

export integrate, to


function _sparse!(I, J, V, m, n)
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


function integrate(func::OptimizedEvaluable, domain, quadrule)
    data = zeros(eltype(func), size(func))
    (pts, wts) = quadrule
    for element in domain
        for (pt, wt) in zip(pts, wts)
            data .+= func(element, pt) .* wt
        end
    end
    data
end


function integrate(func::OptimizedSparseEvaluable, domain, quadrule)
    @assert ndims(func) == 2
    nelems = length(domain)
    nentries = nnz(func)

    I = Vector{Int}(undef, nentries * nelems)
    J = Vector{Int}(undef, nentries * nelems)
    V = zeros(eltype(func), nentries * nelems)
    (pts, wts) = quadrule

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


function _integrate(block::OptimizedBlockEvaluable, domain, quadrule, I, J, V)
    (pts, wts) = quadrule
    for (i, element) in enumerate(domain)
        I[:,:,i] .= block.indices[1](element, nothing)
        J[:,:,i] .= block.indices[2](element, nothing)
        for (pt, wt) in zip(pts, wts)
            V[:,:,i] .+= block.data(element, pt) .* wt
        end
    end
end


end # module
