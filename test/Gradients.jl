@testset "LocalPoint" begin
    Random.seed!(201906191820)
    geom = local_point(2)
    element = Element(2)

    func = optimize(grad(geom, geom))
    @test func(element, @SVector rand(2)) ≈ [1 0; 0 1]

    func = optimize(grad(grad(geom, geom), geom))
    @test func(element, @SVector rand(2)) ≈ zeros(2,2,2)
end


@testset "GlobalPoint" begin
    Random.seed!(201906191820)
    geom = local_point(2)
    element = Element(Shift(@SVector rand(2)))

    func = optimize(grad(global_point(2), geom))
    @test func(element, @SVector rand(2)) ≈ [1 0; 0 1]

    func = optimize(grad(grad(global_point(2), geom), geom))
    @test func(element, @SVector rand(2)) ≈ zeros(2,2,2)
end


@testset "GetIndex" begin
    Random.seed!(201906231109)
    element = Element(3)
    quadpt = @SVector [1.0, 2.0, 3.0]
    geom = local_point(3)

    ufunc = Monomials(geom, 4)

    res = optimize(grad(ufunc[1,:], geom))(element, quadpt)
    @test res ≈ [0 0 0; 1 0 0; 2 0 0; 3 0 0; 4 0 0]

    res = optimize(grad(ufunc[2,:], geom))(element, quadpt)
    @test res ≈ [0 0 0; 0 1 0; 0 4 0; 0 12 0; 0 32 0]

    res = optimize(grad(ufunc[3,:], geom))(element, quadpt)
    @test res ≈ [0 0 0; 0 0 1; 0 0 6; 0 0 27; 0 0 108]
end


@testset "Monomials" begin
    Random.seed!(201906201133)
    element = Element(3)
    quadpt = @SVector [1.0, 2.0, 3.0]
    geom = local_point(3)

    ufunc = grad(Monomials(geom, 4), geom)
    res = optimize(ufunc)(element, quadpt)
    @test res[:,:,1] ≈ [0 1 2 3 4; 0 0 0 0 0; 0 0 0 0 0]
    @test res[:,:,2] ≈ [0 0 0 0 0; 0 1 4 12 32; 0 0 0 0 0]
    @test res[:,:,3] ≈ [0 0 0 0 0; 0 0 0 0 0; 0 1 6 27 108]

    ufunc = grad(ufunc, geom)
    res = optimize(ufunc)(element, quadpt)
    @test res[:,:,1,1] ≈ [0 0 2 6 12; 0 0 0 0 0; 0 0 0 0 0]
    @test res[:,:,2,2] ≈ [0 0 0 0 0; 0 0 2 12 48; 0 0 0 0 0]
    @test res[:,:,3,3] ≈ [0 0 0 0 0; 0 0 0 0 0; 0 0 2 18 108]
    @test (
        res[:,:,1,2] ≈ res[:,:,1,3] ≈ res[:,:,2,1] ≈
        res[:,:,2,3] ≈ res[:,:,3,1] ≈ res[:,:,3,2] ≈
        zeros(3,5)
    )
end


@testset "Multiply" begin
    Random.seed!(201906201233)
    element = Element(3)
    quadpt = @SVector [1.0, 2.0, 3.0]
    geom = local_point(3)

    ufunc = grad(reshape(geom, 1, :) .* geom, geom)
    res = optimize(ufunc)(element, quadpt)
    @test res[:,:,1] ≈ [2 2 3; 2 0 0; 3 0 0]
    @test res[:,:,2] ≈ [0 1 0; 1 4 3; 0 3 0]
    @test res[:,:,3] ≈ [0 0 1; 0 0 2; 1 2 6]

    ufunc = grad(ufunc, geom)
    res = optimize(ufunc)(element, quadpt)
    @test res[:,:,1,1] ≈ [2 0 0; 0 0 0; 0 0 0]
    @test res[:,:,1,2] ≈ [0 1 0; 1 0 0; 0 0 0]
    @test res[:,:,1,3] ≈ [0 0 1; 0 0 0; 1 0 0]
    @test res[:,:,2,1] ≈ [0 1 0; 1 0 0; 0 0 0]
    @test res[:,:,2,2] ≈ [0 0 0; 0 2 0; 0 0 0]
    @test res[:,:,2,3] ≈ [0 0 0; 0 0 1; 0 1 0]
    @test res[:,:,3,1] ≈ [0 0 1; 0 0 0; 1 0 0]
    @test res[:,:,3,2] ≈ [0 0 0; 0 0 1; 0 1 0]
    @test res[:,:,3,3] ≈ [0 0 0; 0 0 0; 0 0 2]
end
