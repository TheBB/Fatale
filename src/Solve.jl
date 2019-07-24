module Solve

import ..Utils: exterior
import ..Integrate: integrate

export project


_colsupp(mx) = findall(!iszero, diff(mx.colptr))

function project(func, basis, domain, quadrule)
    cons = Vector{Union{Float64,Missing}}(missing, size(basis, 1))
    project!(func, basis, domain, quadrule, cons)
end

function project!(func, basis, domain, quadrule, cons)
    mx = integrate(exterior(basis), domain, quadrule)
    rhs = integrate(func .* basis, domain, quadrule)
    solve!(cons, mx, rhs, _colsupp(mx))
    cons
end


function solve!(lhs, mx, rhs, I)
    lhs[I] .= mx[I,I] \ rhs[I]
end

end
