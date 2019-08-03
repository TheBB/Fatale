@testset "LocalPoint" begin
    Random.seed!(201906141219)
    func = optimize(local_point(2))

    element = Element(shift(@SVector rand(2)))
    quadpt = @SVector rand(2)
    @test @inferred(func(element, quadpt)) == quadpt

    sub = SubElement(updim(Val(2), 1, 5.0), element)
    quadpt = @SVector rand(1)
    @test @inferred(func(sub, quadpt)) == [5.0, quadpt[1]]

    sub = SubElement(updim(Val(2), 2, 5.0), element)
    quadpt = @SVector rand(1)
    @test @inferred(func(sub, quadpt)) == [quadpt[1], 5.0]

    @noallocs begin
        func = optimize(local_point(2))
        element = Element(shift(@SVector rand(2)))
        sub = SubElement(updim(Val(2), 2, 5.0), element)
        quadpt = @SVector rand(1)
        @bench $func($sub, $quadpt)
    end
end


@testset "GlobalPoint" begin
    Random.seed!(201906141243)
    func = optimize(global_point(2))
    offset = @SVector rand(2)

    element = Element(shift(offset))
    quadpt = @SVector rand(2)
    @test @inferred(func(element, quadpt)) ≈ quadpt + offset

    sub = SubElement(updim(Val(2), 1, 4.0), element)
    quadpt = @SVector rand(1)
    @test @inferred(func(sub, quadpt)) ≈ [4.0, quadpt[1]] + offset

    sub = SubElement(updim(Val(2), 2, 4.0), element)
    quadpt = @SVector rand(1)
    @test @inferred(func(sub, quadpt)) ≈ [quadpt[1], 4.0] + offset

    @noallocs begin
        func = optimize(global_point(2))
        element = Element(shift(@SVector rand(2)))
        sub = SubElement(updim(Val(2), 2, 5.0), element)
        quadpt = @SVector rand(1)
        @bench $func($sub, $quadpt)
    end
end


@testset "Add" begin
    Random.seed!(201906191836)

    arr1 = @SArray rand(1,2,3)
    arr2 = @SArray rand(2,2)
    arr3 = @SArray rand(2,1,3)

    func = .+(DummyConstant(arr1), DummyConstant(arr2), DummyConstant(arr3))
    @test size(func) == (2, 2, 3)
    func = optimize(func)

    @test @inferred(func(nothing, nothing)) ≈ .+(arr1, arr2, arr3)

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end


@testset "Contract" begin
    Random.seed!(201906171327)

    mx1 = @SMatrix rand(2,3)
    mx2 = @SMatrix rand(3,2)
    func = optimize(Contract(
        [DummyConstant(mx1), DummyConstant(mx2)],
        [[1,2], [2,3]], [1,3]
    ))
    @test @inferred(func(nothing, nothing)) ≈ mx1 * mx2

    func = optimize(DummyConstant(mx1) * DummyConstant(mx2))
    @test @inferred(func(nothing, nothing)) ≈ mx1 * mx2

    mx3 = @SMatrix rand(2,8)
    func = optimize(Contract(
        [DummyConstant(mx1), DummyConstant(mx2), DummyConstant(mx3)],
        [[1,2], [2,3], [3,4]], [1,4]
    ))
    @test @inferred(func(nothing, nothing)) ≈ mx1 * mx2 * mx3

    func = optimize(Contract(
        [DummyConstant(mx1), DummyConstant(mx2), DummyConstant(mx3')],
        [[10,30], [30,90], [71,10]], [71,90]
    ))
    @test @inferred(func(nothing, nothing)) ≈ mx3' * mx1 * mx2

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end


@testset "Constant" begin
    Random.seed!(201906141547)
    data = @SArray rand(3,4,5)
    func = optimize(Constant(data))
    @test @inferred(func(nothing, nothing)) == data

    @noallocs begin
        data = @SArray rand(3,4,5)
        func = optimize(Constant(data))
        @bench $func(nothing, nothing)
    end
end


@testset "ElementIntegral" begin
    func = optimize(ElementIntegral(global_point(2)))
    quadrule = (
        [SVector(0.25, 0.25), SVector(0.75, 0.25), SVector(0.25, 0.75), SVector(0.75, 0.75)],
        [0.25, 0.25, 0.25, 0.25]
    )

    element = Element(Empty(2))
    @test @inferred(func((quadrule=quadrule, element=element))) ≈ [0.5, 0.5]

    element = Element(shift(SVector(2.0, 3.0)))
    @test @inferred(func((quadrule=quadrule, element=element))) ≈ [2.5, 3.5]
end


@testset "GetIndex" begin
    Random.seed!(201906221119)
    data = @SArray rand(3,5,7)

    ufunc = DummyConstant(data)[1, :, :]
    @test size(ufunc) == (5, 7)
    func = optimize(ufunc)
    res = @inferred(func(nothing, nothing))
    @test res == data[1, :, :]

    @noallocs begin
        @bench $func(nothing, nothing)
    end

    ufunc = DummyConstant(data)[3, :, 4]
    @test size(ufunc) == (5,)
    func = optimize(ufunc)
    res = @inferred(func(nothing, nothing))
    @test res == data[3, :, 4]

    @noallocs begin
        @bench $func(nothing, nothing)
    end

    ufunc = DummyConstant(data)[:, 2, :]
    @test size(ufunc) == (3, 7)
    func = optimize(ufunc)
    res = @inferred(func(nothing, nothing))
    @test res == data[:, 2, :]

    @noallocs begin
        @bench $func(nothing, nothing)
    end

    ufunc = DummyConstant(data)[1:end-1, 3, :]
    @test size(ufunc) == (2, 7)
    func = optimize(ufunc)
    res = @inferred(func(nothing, nothing))
    @test res == data[1:end-1, 3, :]

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end


@testset "Inv" begin
    Random.seed!(201906191343)

    data = @SArray rand(1,1)
    func = optimize(inv(DummyConstant(data)))
    @test @inferred(func(nothing, nothing)) ≈ inv(data)

    data = @SArray rand(2,2)
    func = optimize(inv(DummyConstant(data)))
    @test @inferred(func(nothing, nothing)) ≈ inv(data)

    data = @SArray rand(3,3)
    func = optimize(inv(DummyConstant(data)))
    @test @inferred(func(nothing, nothing)) ≈ inv(data)

    @noallocs begin
        data = @SArray rand(3,3)
        func = optimize(inv(DummyConstant(data)))
        @bench $func(nothing, nothing)
    end
end


@testset "Monomials" begin
    Random.seed!(201906141455)
    element = Element(3)
    quadpt = @SVector [1.0, 2.0, 3.0]

    func = optimize(Monomials(local_point(3), 4))
    @test @inferred(func(element, quadpt)) ≈ [1 1 1 1 1; 1 2 4 8 16; 1 3 9 27 81]

    func = optimize(Monomials(local_point(3), 4, 2))
    @test @inferred(func(element, quadpt)) ≈ [0 0 1 1 1 1 1; 0 0 1 2 4 8 16; 0 0 1 3 9 27 81]

    @noallocs begin
        func = optimize(Monomials(local_point(3), 4))
        element = Element(3)
        quadpt = @SVector [1.0, 2.0, 3.0]
        @bench $func($element, $quadpt)
    end
end


@testset "Multiply" begin
    Random.seed!(201906191850)

    arr1 = @SArray rand(1,2,3)
    arr2 = @SArray rand(2,2)
    arr3 = @SArray rand(2,1,3)

    func = .*(DummyConstant(arr1), DummyConstant(arr2), DummyConstant(arr3))
    @test size(func) == (2, 2, 3)
    func = optimize(func)

    @test @inferred(func(nothing, nothing)) ≈ .*(arr1, arr2, arr3)

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end


@testset "Negate" begin
    Random.seed!(201906192304)
    data = @SArray rand(5,2)
    func = optimize(-DummyConstant(data))
    @test @inferred(func(nothing, nothing)) == -data

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end


@testset "OneTo" begin
    func = optimize(Evaluables.OneTo(9))
    @test @inferred(func(nothing, nothing)) == SOneTo(9)

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end


@testset "PermuteDims" begin
    Random.seed!(201907181354)
    data = @SArray rand(3,5,7)

    func = optimize(permutedims(DummyConstant(data), (2, 1, 3)))
    @test size(func) == (5, 3, 7)
    @test @inferred(func(nothing, nothing)) == permutedims(data, (2, 1, 3))

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end


@testset "Power" begin
    Random.seed!(201907231411)
    data = @SArray rand(3,5,7)

    func = optimize(DummyConstant(data) .^ 3.1)
    @test @inferred(func(nothing, nothing)) ≈ data .^ 3.1

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end


@testset "Reciprocal" begin
    Random.seed!(201907231420)
    data = @SArray rand(3,5,7)

    func = optimize(Evaluables.Reciprocal(DummyConstant(data)))
    @test @inferred(func(nothing, nothing)) ≈ 1 ./ data

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end


@testset "Reshape" begin
    Random.seed!(201906201022)
    arr = @SArray rand(5, 3, 7)

    func = optimize(reshape(DummyConstant(arr), 3, 7, 5))
    @test @inferred(func(nothing, nothing)) == reshape(arr, 3, 7, 5)

    func = optimize(reshape(DummyConstant(arr), 15, 7))
    @test @inferred(func(nothing, nothing)) == reshape(arr, 15, 7)

    func = optimize(reshape(DummyConstant(arr), 3, :))
    @test @inferred(func(nothing, nothing)) == reshape(arr, 3, 35)

    @noallocs begin
        func = optimize(reshape(DummyConstant(arr), 3, 7, 5))
        @bench $func(nothing, nothing)
    end
end


@testset "Sqrt" begin
    Random.seed!(201907231424)
    data = @SArray rand(3,5,7)

    func = optimize(Evaluables.Sqrt(DummyConstant(data)))
    @test @inferred(func(nothing, nothing)) ≈ sqrt.(data)

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end


@testset "Sum" begin
    Random.seed!(201907120933)
    arr = @SArray rand(2,2,2)

    func = optimize(sum(DummyConstant(arr); dims=(1,)))
    @test @inferred(func(nothing, nothing)) ≈ sum(arr; dims=1)

    @noallocs begin
        @bench $func(nothing, nothing)
    end

    func = optimize(sum(DummyConstant(arr); dims=(1,2)))
    @test @inferred(func(nothing, nothing)) ≈ sum(Array(arr); dims=(1,2))

    @noallocs begin
        @bench $func(nothing, nothing)
    end

    func = optimize(sum(DummyConstant(arr); dims=(1,2), collapse=true))
    @test @inferred(func(nothing, nothing)) ≈ dropdims(sum(Array(arr); dims=(1,2)); dims=(1,2))

    @noallocs begin
        @bench $func(nothing, nothing)
    end

    func = optimize(sum(DummyConstant(arr); collapse=true))
    @test @inferred(func(nothing, nothing)) ≈ Scalar(sum(arr))
end


@testset "Zeros" begin
    Random.seed!(201906191751)
    func = optimize(Zeros(Float64, 3, 5, 7))
    @test @inferred(func(nothing, nothing)) == zeros(Float64, 3, 5, 7)

    @noallocs begin
        @bench $func(nothing, nothing)
    end
end
