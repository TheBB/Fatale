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
