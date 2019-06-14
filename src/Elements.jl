module Elements

import Base: @_inline_meta
using Base.Iterators: product
using FastGaussQuadrature
using StaticArrays

using ..Transforms
using ..Utils

export ReferenceElement, Simplex, Tensor
export Element, SubElement
export quadrule
export loctrans, globtrans


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
    Element{D, I, T}

A full D-dimensional element with index type I and transform type T.
"""
struct Element{D, I, Trf<:Transform} <: AbstractElement{D}
    index :: I
    transform :: Trf
end

# Convenience constructors
Element(trf::Transform) = Element(nothing, trf)
Element(D::Int) = Element(Empty(D))

@generated function Element(index, trf::Transform)
    @assert fromdims(trf) == todims(trf)
    quote
        @_inline_meta
        Element{$(todims(trf)), $index, $trf}(index, trf)
    end
end


"""
    SubElement{D, I, T, P}

A D-dimensional element that is part of a higher-dimensional element
of type P.
"""
struct SubElement{D, I, Trf<:Transform, Parent<:AbstractElement} <: AbstractElement{D}
    index :: I
    transform :: Trf
    parent :: Parent
end

# Convenience constructor with a no-op index
SubElement(trf, parent) = SubElement(nothing, trf, parent)

@generated function SubElement(index, trf::Transform, parent::Element)
    @assert todims(trf) == ndims(parent)
    quote
        @_inline_meta
        SubElement{$(fromdims(trf)), $index, $trf, $parent}(index, trf, parent)
    end
end


"""
    loctrans(::AbstractElement) :: Transform

The transformation necessary to bring quadrature points into the fully
realized parameter space of the master element. This is usually a
no-op for Elements, and a chain of Updims for SubElements
(i.e. boundary elements).
"""
function loctrans(::AbstractElement) end
@inline loctrans(::Element{D}) where D = Empty{D,Float64}()
@inline loctrans(self::SubElement) = Chain(self.transform, loctrans(self.parent))


"""
    globtrans(::AbstractElement) :: Transform

The transformation necessary to bring a fully realized parameter space
point into 'physical' space. This usually only depends on the master
element, and is equal for all subelements.
"""
function globtrans(::AbstractElement) end
@inline globtrans(self::Element) = self.transform
@inline globtrans(self::SubElement) = globtrans(self.parent)


end # module
