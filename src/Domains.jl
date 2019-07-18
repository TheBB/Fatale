module Domains

import StaticArrays: SVector, SMatrix

using ..Transforms
using ..Elements
using ..Evaluables
using ..Utils

export Boundary
export Lagrange
export quadrule
export global_basis, local_basis, doflist
export TensorDomain


# A domain is nothing but an AbstractArray whose eltype is a subtype
# of AbstractElement.


# Basis types

abstract type Basis end
struct Lagrange <: Basis end


"""
    reference(::Domain)

Return the reference element for this domain.
"""
reference(::AbstractArray{E}) where E = Elements.reference(E)


"""
    quadrule(::Domain, args...)

Generate a quadrature rule suitable for elements of the domain.
"""
quadrule(self, args...) where E = Elements.quadrule(reference(self), args...)


# Standard boundary interface

"""
    Boundary(::Domain)

A dummy object suitable for obtaining domain boundaries. A *Boundary*
object should be indexed using only Colon, 1 and end. For example,

To obtain the left edge of a two-dimensional domain:

    Boundary(domain)[1, :]

To obtain the top face of a three-dimensional domain:

    Boundary(domain)[:, :, end]

Boundary objects should not be used for any other purpose, even though
they will type check as actual domains.
"""
struct Boundary{Elt,N} <: AbstractArray{Elt,N}
    domain :: AbstractArray{Elt,N}
end

# Add one because we're not counting elements but 'meshlines'
Base.size(self::Boundary) = map(x->x+1, size(self.domain))

@inline Base.IndexStyle(::Type{Boundary}) = IndexCartesian()
function Base.getindex(self::Boundary, I::Vararg{Union{Int,Colon}})
    @assert all(zip(I, size(self))) do ((ix, sz))
        ix isa Colon || ix == 1 || ix == sz
    end
    all(ix isa Colon for ix in I) && return self.domain
    boundary(self.domain, I)
end

"""
    boundary(domain, I::Tuple{Vararg{Union{Int,Colon}}})

Create a boundary for *domain* based on the given index. Specific
domains may override this function, but users should create boundaries
using the *Boundary* dummy object.

The default implementation creates a *BoundaryView* object with a
fixed transform, obtained by calling

    boundary_trf(domain, I)
"""
function boundary(domain, I)
    all(ix isa Colon for ix in I) && return domain
    trf = boundary_trf(domain, I)
    I = map(I) do ix
        ix isa Colon && return ix
        ix > 1 ? ix-1 : ix
    end
    BoundaryView(view(domain, I...), trf)
end

"""
    BoundaryView(parent, transform)

A BoundaryView functions as a domain whose elements are sub-elements
of those of *parent*. The given transform is composed with the local
transform of the parent elements.
"""
struct BoundaryView{P,T,E,N} <: AbstractArray{E,N}
    parent :: P
    transform :: T

    function BoundaryView(parent::P, transform::T) where {E,N,P<:AbstractArray{E,N},T}
        eltype = SubElement{fromdims(T), T, E}
        new{P,T,eltype,N}(parent, transform)
    end
end

Base.IndexStyle(::Type{<:BoundaryView{P}}) where P = IndexStyle(P)
@inline Base.size(self::BoundaryView) = size(self.parent)
@inline Base.getindex(self::BoundaryView, I...) = SubElement(self.transform, self.parent[I...])


"""
    TensorElement{D}(index...)

A standard D-dimensional tensorial element with integer indices.
"""
struct TensorElement{D} <: AbstractElement{D}
    index :: NTuple{D, Int}
end

@inline Elements.globtrans(self::TensorElement{D}) where D = shift(SVector{D,Float64}(self.index) - 1.0)
@inline Elements.index(self::TensorElement) = SVector(self.index)
Elements.reference(::Type{TensorElement{D}}) where D = TensorReference(SimplexReference{1}(), D)
Elements.reference(::Type{<:SubElement{D,T,<:TensorElement}}) where {D,T} = TensorReference(SimplexReference{1}(), D)


"""
    TensorDomain(shape...)

Construct a tensorial domain with a given shape. Its elements are of
type TensorElement.
"""
struct TensorDomain{D} <: AbstractArray{TensorElement{D}, D}
    size :: NTuple{D, Int}
    TensorDomain(shape::Int...) = new{length(shape)}(shape)
end

@inline Base.size(self::TensorDomain) = self.size
@inline Base.IndexStyle(::Type{<:TensorDomain}) = IndexCartesian()
@inline function Base.getindex(self::TensorDomain{D}, I::Vararg{Int,D}) where D
    @boundscheck checkbounds(self, I...)
    TensorElement(I)
end

# A boundary transform on a tensor domain is a composition of Updim
# transforms
function boundary_trf(self::TensorDomain{D}, I) where D
    trf = Empty(D)
    d = D
    for (i, ix) in reverse(collect(enumerate(I)))
        ix isa Colon && continue
        trf = trf ∘ updim(Val(d), i, ix == 1 ? 0.0 : 1.0, ix != 1)
        d -= 1
    end
    trf
end


"""
    local_basis(::TensorDomain, Lagrange, degree::Int)

Create an evaluable computing the shape functions for a Lagrangian
basis.
"""
function local_basis(self::TensorDomain{D}, ::Type{Lagrange}, degree) where D
    # Generate D single-dimensional Lagrangian bases
    poly = Monomials(local_point(D), degree)
    coeffs = inv(range(0, 1, length=degree+1) .^ reshape(0:degree, 1, :))
    coeffs = SMatrix{degree+1, degree+1}(coeffs)
    basis1d = poly * Constant(coeffs)

    # Reshape and form an outer product
    factors = (insertaxis(basis1d[i,:]; left=i-1) for i in 1:D)
    return reshape(outer(factors...), :)
end

"""
    doflist(::TensorDomain, Lagrange, degree::Int)

Create an evaluable computing the degree-of-freedom indices for a
Lagrangian basis.
"""
function doflist(self::TensorDomain{D}, ::Type{Lagrange}, degree) where D
    strides = cumprod(collect(Int, size(self)) * degree .+ 1)
    strides = [1, strides[1:end-1]...]
    rootindex = sum(element_index(D) .* Constant(degree * strides); collapse=true)

    offsets = 0 : degree
    for s in strides[2:end]
        offsets = (offsets .+ s * reshape(0:degree, 1, :))[:]
    end

    list = rootindex + (offsets .+ 1 .- sum(degree * strides))
    max = prod(size(self) .* degree .+ 1)
    (list, max)
end


"""
    global_basis(::Domain, type, args...)

Create a basis of type *type* on the given domain. The rest of the
arguments depend on the domain and the type. The return value is an
evaluable object.
"""
function global_basis(args...)
    loc = local_basis(args...)
    (indices, maxindex) = doflist(args...)
    Inflate(loc, indices, maxindex)
end


end # module
