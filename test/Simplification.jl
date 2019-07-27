@testset "Add" begin
    a1 = DummyConstant(@SArray [1,0,0])
    a2 = DummyConstant(@SArray [2,0,0])
    c1 = Constant(@SArray [4,0,0])
    c2 = Constant(@SArray [5,0,0])
    c3 = Constant(@SArray [6,0,0])
    c4 = Evaluables.OneTo(3)
    c5 = Evaluables.FUnitRange(5,7)

    q = c1 .+ c2 .+ c3
    @test q isa Constant
    @test Evaluables.valueof(q) == [15, 0, 0]

    q = a1 .+ c2
    @test q isa Evaluables.Add
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) == [5, 0, 0]

    q = a1 .+ c2 .+ c3
    @test q isa Evaluables.Add
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) == [11, 0, 0]

    q = c1 .+ a1 .+ c2 .+ c3
    @test q isa Evaluables.Add
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) == [15, 0, 0]

    q = c1 .+ a1 .+ c2 .+ c3 .+ a2
    @test q isa Evaluables.Add
    @test length(q.args) == 3
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test q.args[3] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) == [15, 0, 0]

    q1 = c1 .+ a1
    q2 = c2 .+ a2
    q = q1 .+ q2
    @test q isa Evaluables.Add
    @test length(q.args) == 3
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test q.args[3] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) == [9, 0, 0]

    q = c1 .+ c4
    @test q isa Constant
    @test Evaluables.valueof(q) == [5, 2, 3]

    q = c1 .+ c5
    @test q isa Constant
    @test Evaluables.valueof(q) == [9, 6, 7]

    q = c4 .+ c5
    @test q isa Constant
    @test Evaluables.valueof(q) == [6, 8, 10]
end


@testset "Contract" begin
    Random.seed!(201707251002)

    d1 = @SArray rand(2,2)
    d2 = @SArray rand(2,2)
    d3 = @SArray rand(2,2)

    a1 = DummyConstant(@SArray rand(2,2))
    a2 = DummyConstant(@SArray rand(2,2))
    c1 = Constant(d1)
    c2 = Constant(d2)
    c3 = Constant(d3)

    q = c1 * c2 * c3
    @test q isa Constant
    @test Evaluables.valueof(q) == d1 * d2 * d3

    q = a1 * c2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) == d2

    q = a1 * c2 * c3
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) == d2 * d3

    q1 = a1 * c1
    q2 = a2 * c2
    q = q1 * q2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test q.args[3] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) ≈ reshape(d1, 2, 2, 1, 1) .* reshape(d2, 1, 1, 2, 2)

    z = Zeros(2,5,7)
    q = a1 * z
    @test q == Zeros(2,5,7)

    z = Zeros(5,1,2)
    q = z * a2
    @test q == Zeros(5,1,2)

    i1 = Inflate(DummyConstant(@SArray rand(2,2)), DummyConstant(@SArray [4,5]), 9, 2)
    i2 = Inflate(DummyConstant(@SArray rand(4,2)), DummyConstant(@SArray [4,5]), 7, 2)
    a1 = DummyConstant(@SArray rand(9,4))

    # Contraction along inflated axis
    q = i1 * a1
    @test q isa Contract
    @test size(q) == (2,4)
    map(noinflate, q)

    # Contraction along non-inflated axis
    q = a1 * i2
    @test q isa Inflate
    @test q.arg isa Contract
    @test q.newsize == 7
    @test q.axis == 2
    @test size(q) == (9,7)
    map(noinflate, q.arg)
end


@testset "Multiply" begin
    a1 = DummyConstant(@SArray [1,0,0])
    a2 = DummyConstant(@SArray [2,0,0])
    c1 = Constant(@SArray [4,0,0])
    c2 = Constant(@SArray [5,0,0])
    c3 = Constant(@SArray [6,0,0])
    c4 = Evaluables.OneTo(3)
    c5 = Evaluables.FUnitRange(5,7)

    q = c1 .* c2 .* c3
    @test q isa Constant
    @test Evaluables.valueof(q) == [120, 0, 0]

    q = a1 .* c2
    @test q isa Evaluables.Multiply
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) == [5, 0, 0]

    q = a1 .* c2 .* c3
    @test q isa Evaluables.Multiply
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) == [30, 0, 0]

    q = c1 .* a1 .* c2 .* c3
    @test q isa Evaluables.Multiply
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) == [120, 0, 0]

    q = c1 .* a1 .* c2 .* c3 .* a2
    @test q isa Evaluables.Multiply
    @test length(q.args) == 3
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test q.args[3] isa DummyConstant
    @test Evaluables.valueof(q.args[1]) == [120, 0, 0]

    q = c1 .* c4
    @test q isa Constant
    @test Evaluables.valueof(q) == [4, 0, 0]

    q = c1 .* c5
    @test q isa Constant
    @test Evaluables.valueof(q) == [20, 0, 0]

    q = c4 .* c5
    @test q isa Constant
    @test Evaluables.valueof(q) == [5, 12, 21]
end
