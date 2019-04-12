@testset "Empty" begin
    Random.seed!(201904121653)

    trf = Empty(4)
    initial = @SVector rand(4)
    @test trf(initial) == initial

    grad = SMatrix{4,4,Float64}(I)
    (point, grad) = trf(initial, grad)
    @test point == initial
    @test grad == I

    @noallocs begin
        trf = Empty(4)
        initial = @SVector rand(4)
        grad = SMatrix{4,4,Float64}(I)
        @bench begin
            $trf($initial)
            $trf($initial, $grad)
        end
    end
end


@testset "Shift" begin
    Random.seed!(201904121652)

    trf = Shift(@SVector rand(4))
    initial = @SVector rand(4)
    @test trf(initial) == initial + trf.offset

    grad = SMatrix{4,4,Float64}(I)
    (point, grad) = trf(initial, grad)
    @test point == initial + trf.offset
    @test grad == I

    @noallocs begin
        trf = Shift(@SVector rand(4))
        initial = @SVector rand(4)
        grad = SMatrix{4,4,Float64}(I)
        @bench begin
            $trf($initial)
            $trf($initial, $grad)
        end
    end
end
