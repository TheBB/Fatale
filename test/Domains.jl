@testset "TensorDomain" begin
    domain = TensorDomain(3, 5, 7)
    @test eltype(domain) == Domains.TensorElement{3}
    @test reference(eltype(domain)) == TensorReference(SimplexReference{1}(), 3)
    @test size(domain) == (3, 5, 7)
    @test index(domain[2, 4, 1]) == [2, 4, 1]
    @test loctrans(domain[1, 1, 1]) isa Empty
    @test globtrans(domain[1, 1, 7]) == shift(SVector(0.0, 0.0, 6.0))

    bnd = Boundary(domain)[1, :, :]
    @test eltype(bnd) <: AbstractSubElement{2, Domains.TensorElement{3}}
    @test reference(eltype(bnd)) == TensorReference(SimplexReference{1}(), 2)
    @test size(bnd) == (5, 7)
    @test index(bnd[3, 3]) == [1, 3, 3]
    @test loctrans(bnd[2, 6]) == updim(Val(3), 1, 0.0)
    @test globtrans(bnd[2, 6]) == shift(SVector(0.0, 1.0, 5.0))

    bnd = Boundary(domain)[end, :, :]
    @test eltype(bnd) <: AbstractSubElement{2, Domains.TensorElement{3}}
    @test reference(eltype(bnd)) == TensorReference(SimplexReference{1}(), 2)
    @test size(bnd) == (5, 7)
    @test index(bnd[3, 3]) == [3, 3, 3]
    @test loctrans(bnd[2, 6]) == updim(Val(3), 1, 1.0)
    @test globtrans(bnd[2, 6]) == shift(SVector(2.0, 1.0, 5.0))

    bnd = Boundary(domain)[:, 1, end]
    @test eltype(bnd) <: AbstractSubElement{1, Domains.TensorElement{3}}
    @test reference(eltype(bnd)) == TensorReference(SimplexReference{1}(), 1)
    @test size(bnd) == (3,)
    @test index(bnd[2]) == [2, 1, 7]
    @test loctrans(bnd[1]) == updim(Val(3), 3, 1.0) ∘ updim(Val(2), 2, 0.0)
    @test globtrans(bnd[3]) == shift(SVector(2.0, 0.0, 6.0))
end
