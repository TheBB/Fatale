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
