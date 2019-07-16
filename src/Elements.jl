module Elements

import Base: @_inline_meta
using Base.Iterators: product
using FastGaussQuadrature
using StaticArrays

using ..Transforms
using ..Utils

export ReferenceElement, SimplexReference, TensorReference
export AbstractElement, AbstractSubElement, SubElement
export reference, elementdata, loctrans, globtrans, index


"""
    abstract type ReferenceElement{D}

Represents a D-dimensional reference element.
"""
abstract type ReferenceElement{D} end

Base.ndims(::Type{<:ReferenceElement{D}}) where D = D
Base.ndims(::ReferenceElement{D}) where D = D


"""
    SimplexReference{D}

Represents a D-dimensional simplex reference element (line, triangle,
tetrahedron, etc.)
"""
struct SimplexReference{D} <: ReferenceElement{D} end

# Convenient but type-unstable constructor
SimplexReference(D::Int) = SimplexReference{D}()

function quadrule(::SimplexReference{1}, npts::Int)
    (pts, wts) = gausslegendre(npts)
    rpts = SVector{1,Float64}[SVector((pt+1)/2) for pt in pts]
    (rpts, wts ./ 2)
end


"""
    TensorReference(terms...)

Represents a tensor product reference element, e.g. for D-dimensional
structured meshes use `Tensor(ntuple(_->SimplexReference{1}(), D))`.
"""
struct TensorReference{D, K} <: ReferenceElement{D}
    terms :: K
end

# Generated inner constructors are awkward
@generated function TensorReference(terms::ReferenceElement...)
    dims = sum(ndims(t) for t in terms)
    K = :(Tuple{$(terms...)})
    :(TensorReference{$dims, $K}(terms))
end

function TensorReference(term::ReferenceElement, n::Int)
    TensorReference{n, NTuple{n, typeof(term)}}(ntuple(_->term, n))
end

quadrule(self::TensorReference{D}, npts::Int) where D = quadrule(self, ntuple(_->npts, D))

function quadrule(self::TensorReference{D}, npts::NTuple{D, Int}) where D
    (pts, wts) = zip((quadrule(term, n) for (term, n) in zip(self.terms, npts))...)
    rwts = outer(wts...)
    rpts = (SVector(vcat(p...)) for p in product(pts...))
    (collect(SVector{D, Float64}, rpts), rwts)
end


abstract type AbstractElement{D} end

Base.ndims(::Type{<:AbstractElement{D}}) where D = D
Base.ndims(::AbstractElement{D}) where D = D

"""
    reference(::Type{<:AbstractElement})

Obtain a reference element for a given element type.
"""
reference(::Type{<:AbstractElement}) = nothing


"""
    elementdata(::AbstractElement, ::Val{type}, args...)

Common interface for obtaining element data of a given type. Some
names are reserved:

- :loctrans -> the local parameter transformation
- :globtrans -> the global physical transformation
- :index -> element index (any type)

Others can be freely used. See Fatale.Evaluables.ElementData.
"""
elementdata(::AbstractElement, ::Val, args...) = nothing

# Easier interfaces for the standard names
@inline elementdata(el::AbstractElement, ::Val{:loctrans}) = loctrans(el)
@inline elementdata(el::AbstractElement, ::Val{:globtrans}) = globtrans(el)
@inline elementdata(el::AbstractElement, ::Val{:index}) = index(el)

@inline loctrans(::AbstractElement{D}) where D = Empty{D,Float64}()
@inline globtrans(::AbstractElement) = nothing
@inline index(::AbstractElement) = nothing


abstract type AbstractSubElement{D,P} <: AbstractElement{D} end

reference(::Type{<:AbstractSubElement{D,P}}) where {D,P} = reference(P)
parent(::AbstractSubElement) = nothing

@inline loctrans(self::AbstractSubElement) = Chain(subtrans(self), loctrans(parent(self)))
@inline globtrans(self::AbstractSubElement) = globtrans(parent(self))
@inline index(self::AbstractSubElement) = index(parent(self))


struct SubElement{D,T,P} <: AbstractSubElement{D,P}
    transform :: T
    parent :: P
    SubElement(trf::T, parent::P) where {T, D, P<:AbstractElement{D}} = new{D-1,T,P}(trf, parent)
end

@inline parent(self::SubElement) = self.parent
@inline subtrans(self::SubElement) = self.transform


end # module
