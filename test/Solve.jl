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
