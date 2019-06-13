@testset "Chain" begin
    Random.seed!(201906131608)

    trf = Chain(Updim{1,2}(1.0), Shift(SVector(4.0, 5.0)), Empty(2))
    initial = @SVector rand(1)
    @test trf(initial) == [5.0, initial[1] + 5.0]

    grad = SMatrix{1,1,Float64}(I)
    (point, grad) = trf(initial, grad)
    @test point == [5.0, initial[1] + 5.0]
    @test grad == [0 1; 1 0]

    @noallocs begin
        initial = @SVector rand(1)
        grad = SMatrix{1,1,Float64}(I)
        @bench begin
            trf = Chain(Updim{1,2}(1.0), Shift(SVector(4.0, 5.0)), Empty(2))
            trf($initial)
            trf($initial, $grad)
        end
    end
end


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
        initial = @SVector rand(4)
        grad = SMatrix{4,4,Float64}(I)
        @bench begin
            trf = Empty{4, Float64}()
            trf($initial)
            trf($initial, $grad)
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
        initial = @SVector rand(4)
        grad = SMatrix{4,4,Float64}(I)
        @bench begin
            trf = Shift(@SVector rand(4))
            trf($initial)
            trf($initial, $grad)
        end
    end
end


@testset "Updim" begin
    Random.seed!(201904121710)

    # Dimension 2 -> 3
    initial = @SVector rand(2)
    igrad = SMatrix{2,2}(1.0, 2.0, 3.0, 4.0)

    trf = Updim{1,3}(rand(Float64))
    (point, grad) = trf(initial, igrad)
    @test point == [trf.value, initial...]
    @test grad == [0 0 -2; 1 3 0; 2 4 0]

    trf = Updim{2,3}(rand(Float64))
    (point, grad) = trf(initial, igrad)
    @test point == [initial[1], trf.value, initial[2]]
    @test grad == [1 3 0; 0 0 2; 2 4 0]

    trf = Updim{3,3}(rand(Float64))
    (point, grad) = trf(initial, igrad)
    @test point == [initial..., trf.value]
    @test grad == [1 3 0; 2 4 0; 0 0 -2]

    # Dimension 1 -> 2
    initial = @SVector rand(1)
    igrad = SMatrix{1,1}(3.0)

    trf = Updim{1,2}(rand(Float64))
    (point, grad) = trf(initial, igrad)
    @test point == [trf.value, initial...]
    @test grad == [0 3; 3 0]

    trf = Updim{2,2}(rand(Float64))
    (point, grad) = trf(initial, igrad)
    @test point == [initial..., trf.value]
    @test grad == [3 0; 0 -3]

    # Dimension 0 -> 1
    initial = SVector{0,Float64}()
    igrad = SMatrix{0,0,Float64}()

    trf = Updim{1,1}(rand(Float64))
    (point, grad) = trf(initial, igrad)
    @test point == [trf.value]
    @test grad == ones(1,1)

    @noallocs begin
        initial = @SVector rand(2)
        igrad = SMatrix{2,2}(1.0, 2.0, 3.0, 4.0)
        @bench begin
            trf = Updim{2,3}(rand(Float64))
            trf($initial)
            trf($initial, $igrad)
        end
    end
end
