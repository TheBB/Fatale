module Domains

using StaticArrays

using ..Transforms
using ..Elements
using ..Evaluables
using ..Utils

export Lagrange
export quadrule
export global_basis, local_basis, doflist
export TensorDomain


abstract type Domain{Elt,N} <: AbstractArray{Elt,N} end

quadrule(::Domain{Elt}, args...) where Elt = Elements.quadrule(reference(Elt), args...)


# Basis types

abstract type Basis end
struct Lagrange <: Basis end


# Tensor domains

struct TensorElement{D} <: AbstractElement{D}
    index :: NTuple{D, Int}
end

@inline Elements.globtrans(self::TensorElement{D}) where D = Shift(SVector{D,Float64}(self.index) - 1.0)
@inline Elements.index(self::TensorElement) = SVector(self.index)
Elements.reference(::Type{TensorElement{D}}) where D = TensorReference(SimplexReference{1}(), D)


struct TensorDomain{D} <: Domain{TensorElement{D}, D}
    size :: NTuple{D, Int}
    TensorDomain(shape::Int...) = new{length(shape)}(shape)
end

@inline Base.size(self::TensorDomain) = self.size
@inline Base.IndexStyle(::Type{<:TensorDomain}) = IndexCartesian()
@inline function Base.getindex(self::TensorDomain{D}, I::Vararg{Int,D}) where D
    @boundscheck checkbounds(self, I...)
    TensorElement(I)
end

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

function global_basis(args...)
    loc = local_basis(args...)
    (indices, maxindex) = doflist(args...)
    Inflate(loc, indices, maxindex)
end


end # module
