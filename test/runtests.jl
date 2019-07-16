using BenchmarkTools
using LinearAlgebra
using Random
using SparseArrays
using StaticArrays
using Test

using Fatale.Transforms
using Fatale.Elements
using Fatale.Evaluables
using Fatale.Domains
using Fatale.Integrate
using Fatale.Utils


# Lightweight elements for testing

struct Element{D, T} <: AbstractElement{D}
    transform :: T
end

Element(trf::AbstractTransform) = Element{todims(trf), typeof(trf)}(trf)
Element(D::Int) = Element(Empty(D))
@inline Elements.globtrans(self::Element) = self.transform

struct SubElement{D, T, P} <: AbstractElement{D}
    transform :: T
    parent :: P
end

SubElement(trf, parent) = SubElement{fromdims(trf), typeof(trf), typeof(parent)}(trf, parent)
@inline Elements.loctrans(self::SubElement) = Chain(self.transform, loctrans(self.parent))
@inline Elements.globtrans(self::SubElement) = globtrans(self.parent)


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

@testset "Domains" begin
    include("Domains.jl")
end

@testset "Evaluables" begin
    include("Evaluables.jl")
end

@testset "Gradients" begin
    include("Gradients.jl")
end

@testset "Bases" begin
    include("Bases.jl")
end

@testset "Integrate" begin
    include("Integrate.jl")
end
