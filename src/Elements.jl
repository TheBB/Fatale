module Elements

import Base: @_inline_meta
using Base.Iterators: product
using FastGaussQuadrature
using StaticArrays

using ..Transforms
using ..Utils

export ReferenceElement, Simplex, Tensor, quadrule
export AbstractElement, elementdata


"""
    abstract type ReferenceElement{D}

Represents a D-dimensional reference element.
"""
abstract type ReferenceElement{D} end

Base.ndims(::Type{<:ReferenceElement{D}}) where D = D
Base.ndims(::ReferenceElement{D}) where D = D


"""
    Simplex{D}

Represents a D-dimensional simplex reference element (line, triangle,
tetrahedron, etc.)
"""
struct Simplex{D} <: ReferenceElement{D} end

# Convenient but type-unstable constructor
Simplex(D::Int) = Simplex{D}()

function quadrule(::Simplex{1}, npts::Int)
    (pts, wts) = gausslegendre(npts)
    rpts = SVector{1,Float64}[SVector((pt+1)/2) for pt in pts]
    (rpts, wts ./ 2)
end


"""
    Tensor(terms...)

Represents a tensor product reference element, e.g. for D-dimensional
structured meshes use `Tensor(ntuple(_->Simplex{1}(), D))`.
"""
struct Tensor{D, K} <: ReferenceElement{D}
    terms :: K
end

# Generated inner constructors are awkward
@generated function Tensor(terms::ReferenceElement...)
    dims = sum(ndims(t) for t in terms)
    K = :(Tuple{$(terms...)})
    :(Tensor{$dims, $K}(terms))
end

quadrule(self::Tensor{D}, npts::Int) where D = quadrule(self, ntuple(_->npts, D))

function quadrule(self::Tensor{D}, npts::NTuple{D, Int}) where D
    (pts, wts) = zip((quadrule(term, n) for (term, n) in zip(self.terms, npts))...)
    rwts = vec(outer(wts...))
    rpts = (SVector(vcat(p...)) for p in product(pts...))
    (collect(SVector{D, Float64}, rpts), rwts)
end


abstract type AbstractElement{D} end

Base.ndims(::Type{<:AbstractElement{D}}) where D = D
Base.ndims(::AbstractElement{D}) where D = D


"""
    elementdata(::AbstractElement, ::Val{type}, args...)

Common interface for obtaining element data of a given type. Some
names are reserved:

- :loctrans -> the local parameter transformation
- :globtrans -> the global physical transformation

Others can be freely used. See Fatale.Evaluables.ElementData.
"""
elementdata(::AbstractElement, ::Val, args...) = nothing
elementdata(::AbstractElement{D}, ::Val{:loctrans}) where D = Empty{D,Float64}()


end # module
