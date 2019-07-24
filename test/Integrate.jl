@testset "Lagrange Basis 1D" begin
    domain = TensorDomain(10)
    basis = global_basis(domain, Lagrange, 1)

    res = integrate(optimize(basis), domain, quadrule(domain, 1))
    @test res == [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1] ./ 2
end


@testset "Lagrange Mass 1D" begin
    domain = TensorDomain(2)
    basis = global_basis(domain, Lagrange, 1)
    mass = exterior(basis)

    res = integrate(optimize(mass), domain, quadrule(domain, 2))
    @test nnz(res) == 7
    (I, J, V) = findnz(res)
    @test I == [1, 2, 1, 2, 3, 2, 3]
    @test J == [1, 1, 2, 2, 2, 3, 3]
    @test V ≈ [2, 1, 1, 4, 1, 1, 2] ./ 6
end


@testset "Lagrange Mass 2D" begin
    domain = TensorDomain(2, 2)
    basis = global_basis(domain, Lagrange, 1)
    mass = exterior(basis)

    res = integrate(optimize(mass), domain, quadrule(domain, 3))
    @test nnz(res) == 49
    (I, J, V) = findnz(res)
    @test I == [1, 2, 4, 5, 1, 2, 3, 4, 5, 6, 2, 3, 5, 6, 1, 2, 4, 5, 7, 8, 1, 2, 3, 4,
                5, 6, 7, 8, 9, 2, 3, 5, 6, 8, 9, 4, 5, 7, 8, 4, 5, 6, 7, 8, 9, 5, 6, 8, 9]
    @test J == [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
                5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9]
    @test V ≈ [4, 2, 2, 1, 2, 8, 2, 1, 4, 1, 2, 4, 1, 2, 2, 1, 8, 4, 2, 1, 1, 4, 1, 4,
               16, 4, 1, 4, 1, 1, 2, 4, 8, 1, 2, 2, 1, 4, 2, 1, 4, 1, 2, 8, 2, 1, 2, 2, 4] ./ 36
end
