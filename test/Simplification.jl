@testset "Add" begin
    a1 = DummyConstant(@SArray [1,0,0])
    a2 = DummyConstant(@SArray [2,0,0])
    c1 = Constant(@SArray [4,0,0])
    c2 = Constant(@SArray [5,0,0])
    c3 = Constant(@SArray [6,0,0])

    q = c1 .+ c2 .+ c3
    @test q isa Constant

    q = a1 .+ c2
    @test q isa Evaluables.Add
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant

    q = a1 .+ c2 .+ c3
    @test q isa Evaluables.Add
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant

    q = c1 .+ a1 .+ c2 .+ c3
    @test q isa Evaluables.Add
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant

    q = c1 .+ a1 .+ c2 .+ c3 .+ a2
    @test q isa Evaluables.Add
    @test length(q.args) == 3
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test q.args[3] isa DummyConstant

    q1 = c1 .+ a1
    q2 = c2 .+ a2
    q = q1 .+ q2
    @test q isa Evaluables.Add
    @test length(q.args) == 3
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test q.args[3] isa DummyConstant
end


@testset "Multiply" begin
    a1 = DummyConstant(@SArray [1,0,0])
    a2 = DummyConstant(@SArray [2,0,0])
    c1 = Constant(@SArray [4,0,0])
    c2 = Constant(@SArray [5,0,0])
    c3 = Constant(@SArray [6,0,0])

    q = c1 .* c2 .* c3
    @test q isa Constant

    q = a1 .* c2
    @test q isa Evaluables.Multiply
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant

    q = a1 .* c2 .* c3
    @test q isa Evaluables.Multiply
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant

    q = c1 .* a1 .* c2 .* c3
    @test q isa Evaluables.Multiply
    @test length(q.args) == 2
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant

    q = c1 .* a1 .* c2 .* c3 .* a2
    @test q isa Evaluables.Multiply
    @test length(q.args) == 3
    @test q.args[1] isa Constant
    @test q.args[2] isa DummyConstant
    @test q.args[3] isa DummyConstant
end
