@testset "Affine" begin
    Random.seed!(201907171232)

    mx = @SMatrix rand(3,3)
    vec = @SVector rand(3)
    trf = Affine(mx, vec)

    initial = @SVector rand(3)
    @test trf(initial) == mx * initial + vec

    grad = SMatrix{3,3,Float64}(I)
    coords = trf((point=initial, grad=grad))
    @test coords.point == mx * initial + vec
    @test coords.grad == mx

    @noallocs begin
        mx = @SMatrix rand(3,3)
        vec = @SVector rand(3)
        initial = @SVector rand(3)
        grad = SMatrix{3,3,Float64}(I)
        @bench begin
            trf = Affine($mx, $vec)
            trf($initial)
            trf((point=$initial, grad=$grad))
        end
    end
end


@testset "Chain" begin
    Random.seed!(201906131608)

    trf = Empty(2) ∘ shift(SVector(4.0, 5.0)) ∘ updim(Val(2), 1, 1.0)
    initial = @SVector rand(1)
    @test trf(initial) == [5.0, initial[1] + 5.0]

    grad = SMatrix{1,1,Float64}(I)
    coords = trf((point=initial, grad=grad))
    @test coords.point == [5.0, initial[1] + 5.0]
    @test coords.grad == [0 1; 1 0]

    @noallocs begin
        initial = @SVector rand(1)
        grad = SMatrix{1,1,Float64}(I)
        @bench begin
            trf = Empty(2) ∘ shift(SVector(4.0, 5.0)) ∘ updim(Val(2), 1, 1.0)
            trf($initial)
            trf((point=$initial, grad=$grad))
        end
    end
end


@testset "Empty" begin
    Random.seed!(201904121653)

    trf = Empty(4)
    initial = @SVector rand(4)
    @test trf(initial) == initial

    grad = SMatrix{4,4,Float64}(I)
    coords = trf((point=initial, grad=grad))
    @test coords.point == initial
    @test coords.grad == I

    @noallocs begin
        initial = @SVector rand(4)
        grad = SMatrix{4,4,Float64}(I)
        @bench begin
            trf = Empty{4, Float64}()
            trf($initial)
            trf((point=$initial, grad=$grad))
        end
    end
end


@testset "Shift" begin
    Random.seed!(201904121652)

    offset = @SVector rand(4)
    trf = Transforms.shift(offset)
    initial = @SVector rand(4)
    @test trf(initial) == initial + offset

    grad = SMatrix{4,4,Float64}(I)
    coords = trf((point=initial, grad=grad))
    @test coords.point == initial + offset
    @test coords.grad == I

    @noallocs begin
        initial = @SVector rand(4)
        offset = @SVector rand(4)
        grad = SMatrix{4,4,Float64}(I)
        @bench begin
            trf = Transforms.shift($offset)
            trf($initial)
            trf((point=$initial, grad=$grad))
        end
    end
end


@testset "Updim" begin
    Random.seed!(201904121710)
    value = rand(Float64)

    # Dimension 2 -> 3
    initial = @SVector rand(2)
    igrad = SMatrix{2,2}(1.0, 2.0, 3.0, 4.0)

    trf = updim(Val(3), 1, value)
    coords = trf((point=initial, grad=igrad))
    @test coords.point == [value, initial...]
    @test coords.grad == [0 0 -2; 1 3 0; 2 4 0]

    trf = updim(Val(3), 2, value)
    coords = trf((point=initial, grad=igrad))
    @test coords.point == [initial[1], value, initial[2]]
    @test coords.grad == [1 3 0; 0 0 2; 2 4 0]

    trf = updim(Val(3), 3, value)
    coords = trf((point=initial, grad=igrad))
    @test coords.point == [initial..., value]
    @test coords.grad == [1 3 0; 2 4 0; 0 0 -2]

    # Dimension 1 -> 2
    initial = @SVector rand(1)
    igrad = SMatrix{1,1}(3.0)

    trf = updim(Val(2), 1, value)
    coords = trf((point=initial, grad=igrad))
    @test coords.point == [value, initial...]
    @test coords.grad == [0 3; 3 0]

    trf = updim(Val(2), 2, value)
    coords = trf((point=initial, grad=igrad))
    @test coords.point == [initial..., value]
    @test coords.grad == [3 0; 0 -3]

    # Dimension 0 -> 1
    initial = SVector{0,Float64}()
    igrad = SMatrix{0,0,Float64}()

    trf = updim(Val(1), 1, value)
    coords = trf((point=initial, grad=igrad))
    @test coords.point == [value]
    @test coords.grad == ones(1,1)

    @noallocs begin
        initial = @SVector rand(2)
        igrad = SMatrix{2,2}(1.0, 2.0, 3.0, 4.0)
        value = rand(Float64)
        @bench begin
            trf = updim(Val(3), 2, $value)
            trf($initial)
            trf((point=$initial, grad=$igrad))
        end
    end

end
