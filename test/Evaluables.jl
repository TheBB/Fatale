@testset "LocalPoint" begin
    Random.seed!(201906141219)
    func = optimize(local_point(2))

    element = Element(Shift(@SVector rand(2)))
    quadpt = @SVector rand(2)
    @test func(element, quadpt) == quadpt

    sub = SubElement(Updim{1,2}(5.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt) == [5.0, quadpt[1]]

    sub = SubElement(Updim{2,2}(5.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt) == [quadpt[1], 5.0]

    @noallocs begin
        func = optimize(local_point(2))
        element = Element(Shift(@SVector rand(2)))
        sub = SubElement(Updim{2,2}(5.0), element)
        quadpt = @SVector rand(1)
        @bench $func($sub, $quadpt)
    end
end


@testset "GlobalPoint" begin
    Random.seed!(201906141243)
    func = optimize(global_point(2))
    shift = @SVector rand(2)

    element = Element(Shift(shift))
    quadpt = @SVector rand(2)
    @test func(element, quadpt) ≈ quadpt + shift

    sub = SubElement(Updim{1,2}(4.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt) ≈ [4.0, quadpt[1]] + shift

    sub = SubElement(Updim{2,2}(4.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt) ≈ [quadpt[1], 4.0] + shift

    @noallocs begin
        func = optimize(global_point(2))
        element = Element(Shift(@SVector rand(2)))
        sub = SubElement(Updim{2,2}(5.0), element)
        quadpt = @SVector rand(1)
        @bench $func($sub, $quadpt)
    end
end


@testset "Contract" begin
    Random.seed!(201906171327)

    mx1 = @SMatrix rand(2,3)
    mx2 = @SMatrix rand(3,2)
    func = optimize(Contract(
        (Constant(mx1), Constant(mx2)),
        ((1,2), (2,3)), (1,3)
    ))
    @test func(nothing, nothing) ≈ mx1 * mx2

    mx3 = @SMatrix rand(2,8)
    func = optimize(Contract(
        (Constant(mx1), Constant(mx2), Constant(mx3)),
        ((1,2), (2,3), (3,4)), (1,4)
    ))
    @test func(nothing, nothing) ≈ mx1 * mx2 * mx3

    func = optimize(Contract(
        (Constant(mx1), Constant(mx2), Constant(mx3')),
        ((10,30), (30,90), (71,10)), (71,90)
    ))
    @test func(nothing, nothing) ≈ mx3' * mx1 * mx2

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end


@testset "Constant" begin
    Random.seed!(201906141547)
    data = @SArray rand(3,4,5)
    func = optimize(Constant(data))
    @test func(nothing, nothing) == data

    @noallocs begin
        data = @SArray rand(3,4,5)
        func = optimize(Constant(data))
        @bench $func(nothing, nothing)
    end
end


@testset "Inv" begin
    Random.seed!(201906191343)

    data = @SArray rand(1,1)
    func = optimize(Inv(Constant(data)))
    @test func(nothing, nothing) ≈ inv(data)

    data = @SArray rand(2,2)
    func = optimize(Inv(Constant(data)))
    @test func(nothing, nothing) ≈ inv(data)

    data = @SArray rand(3,3)
    func = optimize(Inv(Constant(data)))
    @test func(nothing, nothing) ≈ inv(data)

    @noallocs begin
        data = @SArray rand(3,3)
        func = optimize(Inv(Constant(data)))
        @bench $func(nothing, nothing)
    end
end


@testset "Monomials" begin
    Random.seed!(201906141455)
    func = optimize(Monomials(local_point(3), 4))
    element = Element(3)
    quadpt = @SVector [1.0, 2.0, 3.0]
    @test func(element, quadpt) ≈ [1 1 1 1 1; 1 2 4 8 16; 1 3 9 27 81]

    @noallocs begin
        func = optimize(Monomials(local_point(3), 4))
        element = Element(3)
        quadpt = @SVector [1.0, 2.0, 3.0]
        @bench $func($element, $quadpt)
    end
end


@testset "Zeros" begin
    Random.seed!(201906191751)
    func = optimize(Zeros(Float64, 3, 5, 7))
    @test func(nothing, nothing) == zeros(Float64, 3, 5, 7)

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end
