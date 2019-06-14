@testset "LocalPoint" begin
    Random.seed!(201906141219)
    func = optimize(localpoint(2))

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
        func = optimize(localpoint(2))
        element = Element(Shift(@SVector rand(2)))
        sub = SubElement(Updim{2,2}(5.0), element)
        quadpt = @SVector rand(1)
        @bench $func($sub, $quadpt)
    end
end


@testset "GlobalPoint" begin
    Random.seed!(201906141243)
    func = optimize(globalpoint(2))
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
        func = optimize(globalpoint(2))
        element = Element(Shift(@SVector rand(2)))
        sub = SubElement(Updim{2,2}(5.0), element)
        quadpt = @SVector rand(1)
        @bench $func($sub, $quadpt)
    end
end
