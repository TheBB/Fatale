using BenchmarkTools
using LinearAlgebra
using Test
using Random
using StaticArrays

using Fatale.Transforms
using Fatale.Elements
using Fatale.Evaluables


# Lightweight elements for testing

struct Element{D, T} <: AbstractElement{D}
    transform :: T
end

Element(trf::Transform) = Element{todims(trf), typeof(trf)}(trf)
Element(D::Int) = Element(Empty(D))
@inline Elements.elementdata(self::Element, ::Val{:globtrans}) = self.transform

struct SubElement{D, T, P} <: AbstractElement{D}
    transform :: T
    parent :: P
end

SubElement(trf, parent) = SubElement{fromdims(trf), typeof(trf), typeof(parent)}(trf, parent)
@inline Elements.elementdata(self::SubElement, ::Val{:loctrans}) =
    Chain(self.transform, elementdata(self.parent, Val(:loctrans)))
@inline Elements.elementdata(self::SubElement, ::Val{:globtrans}) = elementdata(self.parent, Val(:globtrans))


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

@testset "Evaluables" begin
    include("Evaluables.jl")
end
