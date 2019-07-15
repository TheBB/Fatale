@testset "Simplex" begin
    ref = SimplexReference(1)
    @test ndims(ref) == 1

    (pts, wts) = Elements.quadrule(ref, 3)
    @test length(pts) == 3
    @test sum(pts) ≈ [1.5]
    @test sum(wts) ≈ 1
end


@testset "Tensor" begin
    ref = TensorReference(SimplexReference(1), SimplexReference(1))
    @test ndims(ref) == 2

    (pts, wts) = Elements.quadrule(ref, 4)
    @test length(pts) == 16
    @test sum(pts) ≈ [8, 8]
    @test sum(wts) ≈ 1

    (pts, wts) = Elements.quadrule(ref, (2, 3))
    @test length(pts) == 6
    @test sum(pts) ≈ [3, 3]
    @test sum(wts) ≈ 1
end
