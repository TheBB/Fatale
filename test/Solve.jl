@testset "Project Lagrange 1D" begin
    domain = TensorDomain(4)
    basis = global_basis(domain, Lagrange, 1)

    func = global_point(1)[1]
    @test project(func, basis, domain, quadrule(domain, 4)) ≈ [0, 1, 2, 3, 4]

    func = global_point(1)[1] .^ 2
    @test project(func, basis, domain, quadrule(domain, 4)) ≈ [-1, 5, 23, 53, 95] ./ 6
end


@testset "Project Lagrange 2D" begin
    domain = TensorDomain(4, 4)
    basis = global_basis(domain, Lagrange, 1)

    func = global_point(2)[1]
    bnd = Boundary(domain)[:,1]
    res = project(func, basis, bnd, quadrule(bnd, 4))
    @test res[1:5] ≈ [0, 1, 2, 3, 4]
    @test all(ismissing, res[6:end])

    func = global_point(2)[2]
    bnd = Boundary(domain)[1,:]
    res = project(func, basis, bnd, quadrule(bnd, 4))
    @test res[1:5:25] ≈ [0, 1, 2, 3, 4]
    @test all(ismissing, res[setdiff(1:25, 1:5:25)])

    func = global_point(2)[1] .^ 2
    bnd = Boundary(domain)[:,1]
    res = project(func, basis, bnd, quadrule(bnd, 4))
    @test res[1:5] ≈ [-1, 5, 23, 53, 95] ./ 6
    @test all(ismissing, res[6:end])

    func = global_point(2)[2] .^ 2
    bnd = Boundary(domain)[1,:]
    res = project(func, basis, bnd, quadrule(bnd, 4))
    @test res[1:5:25] ≈ [-1, 5, 23, 53, 95] ./ 6
    @test all(ismissing, res[setdiff(1:25, 1:5:25)])
end


@testset "Poisson Lagrange 2D" begin
    # This is a manufactured solution test case
    # with f(x,y) = x^2 + y^2 and only Dirichlet BCs

    domain = TensorDomain(2,2)
    basis = global_basis(domain, Lagrange, 2)

    geom = global_point(2)
    solution = sum(geom .^ 2; collapse=true)

    bc1 = Boundary(domain)[:,1]
    bc2 = Boundary(domain)[:,end]
    bc3 = Boundary(domain)[1,:]
    bc4 = Boundary(domain)[end,:]
    cons = project(solution, basis, bc1, quadrule(bc1, 5))
    project!(solution, basis, bc2, quadrule(bc2, 5), cons)
    project!(solution, basis, bc3, quadrule(bc3, 5), cons)
    project!(solution, basis, bc4, quadrule(bc4, 5), cons)

    @test isapprox_missing(cons, [
        0, 1, 4, 9, 16,
        1, missing, missing, missing, 17,
        4, missing, missing, missing, 20,
        9, missing, missing, missing, 25,
        16, 17, 20, 25, 32
    ] ./ 4)

    laplacian = sum(exterior(grad(basis, geom)); dims=(3,), collapse=true)
    matrix = integrate(laplacian, domain, quadrule(domain, 5))
    rhs = integrate(-4basis, domain, quadrule(domain, 5))
    lhs = solve(matrix, rhs, cons)

    @test lhs ≈ [
        0, 1, 4, 9, 16,
        1, 2, 5, 10, 17,
        4, 5, 8, 13, 20,
        9, 10, 13, 18, 25,
        16, 17, 20, 25, 32
    ] ./ 4
end
