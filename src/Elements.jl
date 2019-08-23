module Elements

using Base.Iterators: product
using FastGaussQuadrature: gausslegendre
using StaticArrays: SVector
using ..Utils: outer
using ..Transforms: Empty

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
    quadrule(::ReferenceElement, args...)

Generate a quadrature rule for a reference element.
"""
function quadrule end


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
    TensorReference(terms::ReferenceElement...)

Represents a tensor product reference element.
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

"""
    TensorReference(term::ReferenceElement, n::Int)

A 'power' reference element, equivalent to
TensorReference(term, term, ....)
"""
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


"""
    AbstractElement{D}

A supertype for all elements of dimension D.

Implementations of AbstractElement should implement one or more of
- Fatale.Elements.reference
- Fatale.Elements.loctrans
- Fatale.Elements.globtrans
- Fatale.Elements.index
"""
abstract type AbstractElement{D} end

Base.ndims(::Type{<:AbstractElement{D}}) where D = D
Base.ndims(::AbstractElement{D}) where D = D


"""
    reference(::Type{<:AbstractElement})

Obtain a reference element for a given element type.
"""
function reference end


"""
    elementdata(::AbstractElement, ::Val{type}, args...)

Common interface for obtaining element data of a given type. Some
names are reserved:

- :loctrans -> the local parameter transformation
- :globtrans -> the global physical transformation
- :index -> element index (any type)

Others can be freely used. See Fatale.Evaluables.ElementData.

To implement these interfaces for a custom element type, it's simpler
to implement the functions
- Fatale.Elements.loctrans
- Fatale.Elements.globtrans
- Fatale.Elements.index
"""
function elementdata end

# Easier interfaces for the standard names
@inline elementdata(el::AbstractElement, ::Val{:loctrans}) = loctrans(el)
@inline elementdata(el::AbstractElement, ::Val{:globtrans}) = globtrans(el)
@inline elementdata(el::AbstractElement, ::Val{:index}) = index(el)

# Standard implementations
@inline loctrans(::AbstractElement{D}) where D = Empty{D,Float64}()
@inline globtrans(::T) where T<:AbstractElement = error("globtrans not implemented for type $T")
@inline index(::T) where T<:AbstractElement = error("index not implemented for type $T")


"""
    AbstractSubElement{D,P} <: AbstractElement{D}

A standard supertype for sub-elements. Implementations of
AbstractSubElement should implement
- Fatale.Elements.parent
- Fatale.Elements.subtrans

An AbstractSubElement inherits its parents global transformation and
index, and its local transformation is the composition of the return
value of *subtrans* and the local transformation of its parent.
"""
abstract type AbstractSubElement{D,P} <: AbstractElement{D} end

"""
    parent(::AbstractSubElement)

Return the parent element of a given sub-element.
"""
function parent end

@inline loctrans(self::AbstractSubElement) = loctrans(parent(self)) ∘ subtrans(self)
@inline globtrans(self::AbstractSubElement) = globtrans(parent(self))
@inline index(self::AbstractSubElement) = index(parent(self))


"""
    SubElement(trf::Transform, parent::AbstractElement)

A concrete implementation of a sub-element that is good enough for
most purposes.
"""
struct SubElement{D,T,P} <: AbstractSubElement{D,P}
    transform :: T
    parent :: P
    SubElement(trf::T, parent::P) where {T, D, P<:AbstractElement{D}} = new{D-1,T,P}(trf, parent)
end

@inline parent(self::SubElement) = self.parent
@inline subtrans(self::SubElement) = self.transform


end # module
