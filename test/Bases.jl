@testset "Lagrange" begin
    domain = TensorDomain(1)
    basis = optimize(local_basis(domain, Lagrange, 1))
    @test @inferred(basis(domain[1], @SVector [0.1])) ≈ [0.9, 0.1]
    @test @inferred(basis(domain[1], @SVector [0.3])) ≈ [0.7, 0.3]

    @noallocs begin
        el = domain[1]
        qpt = @SVector [0.5]
        ws = workspace(basis)
        @bench $basis($ws, $el, $qpt)
    end

    basis = optimize(local_basis(domain, Lagrange, 2))
    @test @inferred(basis(domain[1], @SVector [0.1])) ≈ [0.72, 0.36, -0.08]
    @test @inferred(basis(domain[1], @SVector [0.3])) ≈ [0.28, 0.84, -0.12]

    @noallocs begin
        el = domain[1]
        qpt = @SVector [0.5]
        ws = workspace(basis)
        @bench $basis($ws, $el, $qpt)
    end

    basis = optimize(local_basis(domain, Lagrange, 3))
    @test @inferred(basis(domain[1], @SVector [0.1])) ≈ [0.5355, 0.6885, -0.2835, 0.0595]
    @test @inferred(basis(domain[1], @SVector [0.3])) ≈ [0.0385, 1.0395, -0.0945, 0.0165]

    @noallocs begin
        el = domain[1]
        qpt = @SVector [0.5]
        ws = workspace(basis)
        @bench $basis($ws, $el, $qpt)
    end

    domain = TensorDomain(1, 1)
    basis = optimize(local_basis(domain, Lagrange, 3))
    @test @inferred(basis(domain[1,1], @SVector [0.1, 0.1])) ≈ [
         0.28676025,  0.36869175, -0.15181425,  0.03186225,
         0.36869175,  0.47403225, -0.19518975,  0.04096575,
        -0.15181425, -0.19518975,  0.08037225, -0.01686825,
         0.03186225,  0.04096575, -0.01686825,  0.00354024,
    ]
    @test @inferred(basis(domain[1,1], @SVector [0.1, 0.3])) ≈ [
         0.02061675,  0.02650725, -0.01091475,  0.00229075,
         0.55665225,  0.71569575, -0.29469825,  0.06185025,
        -0.05060475, -0.06506325,  0.02679075, -0.00562275,
         0.00883575,  0.01136025, -0.00467775,  0.00098175,
    ]

    @noallocs begin
        el = domain[1, 1]
        qpt = @SVector [0.5, 0.5]
        ws = workspace(basis)
        @bench $basis($ws, $el, $qpt)
    end
end
