using BenchmarkTools
using LinearAlgebra
using Test
using Random
using StaticArrays

using Fatale.Transforms
using Fatale.Elements


"Check that the result of `expr`, which should be a benchmark, has no
allocations."
macro noallocs(expr)
    quote
        trial = $(esc(expr))
        @test trial.allocs == 0
    end
end


"Run `expr` as a benchmark with just one sample and evaluation."
macro bench(expr)
    :(@benchmark $expr samples=1 evals=1)
end


@testset "Transforms" begin
    include("Transforms.jl")
end

@testset "Elements" begin
    include("Elements.jl")
end
