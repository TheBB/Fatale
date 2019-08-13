using BenchmarkTools

using Fatale.Domains
using Fatale.Evaluables
using Fatale.Integrate
using Fatale.Utils


BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20
const SUITE = BenchmarkGroup()


function _integrate(stuff)
    domain, integrand, quadrule = stuff
    integrate(integrand, domain, quadrule)
end


# Integrate laplacian, 2D Lagrange basis (linear)
function lapl_lagr(d, p)
    nelems = round(Int, 10_000^(1/d))
    domain = TensorDomain(ntuple(_->nelems, d)...)
    basis = global_basis(domain, Lagrange, p)
    geom = global_point(d)
    integrand = sum(exterior(grad(basis, geom)); dims=(1,), collapse=true)
    (domain, integrand, quadrule(domain, 5))
end

SUITE["laplace"] = BenchmarkGroup()
SUITE["laplace"]["1d"] = BenchmarkGroup()
SUITE["laplace"]["1d"]["lagr"] = BenchmarkGroup()
SUITE["laplace"]["1d"]["lagr"]["1"] = @benchmarkable _integrate(stuff) setup=(stuff=lapl_lagr(1,1))
SUITE["laplace"]["1d"]["lagr"]["2"] = @benchmarkable _integrate(stuff) setup=(stuff=lapl_lagr(1,2))
SUITE["laplace"]["1d"]["lagr"]["3"] = @benchmarkable _integrate(stuff) setup=(stuff=lapl_lagr(1,3))
SUITE["laplace"]["2d"] = BenchmarkGroup()
SUITE["laplace"]["2d"]["lagr"] = BenchmarkGroup()
SUITE["laplace"]["2d"]["lagr"]["1"] = @benchmarkable _integrate(stuff) setup=(stuff=lapl_lagr(2,1))
SUITE["laplace"]["2d"]["lagr"]["2"] = @benchmarkable _integrate(stuff) setup=(stuff=lapl_lagr(2,2))
SUITE["laplace"]["2d"]["lagr"]["3"] = @benchmarkable _integrate(stuff) setup=(stuff=lapl_lagr(2,3))
SUITE["laplace"]["3d"] = BenchmarkGroup()
SUITE["laplace"]["3d"]["lagr"] = BenchmarkGroup()
SUITE["laplace"]["3d"]["lagr"]["1"] = @benchmarkable _integrate(stuff) setup=(stuff=lapl_lagr(3,1))
SUITE["laplace"]["3d"]["lagr"]["2"] = @benchmarkable _integrate(stuff) setup=(stuff=lapl_lagr(3,2))
SUITE["laplace"]["3d"]["lagr"]["3"] = @benchmarkable _integrate(stuff) setup=(stuff=lapl_lagr(3,3))
