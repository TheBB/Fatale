module Domains

using StaticArrays

using ..Transforms
using ..Elements
using ..Evaluables

export Lagrange
export local_basis
export TensorDomain


abstract type Domain{Elt,Ref,N} <: AbstractArray{Elt,N} end


# Basis types

abstract type Basis end
struct Lagrange <: Basis end


# Tensor domains

struct TensorElement{D} <: AbstractElement{D}
    index :: NTuple{D, Int}
end

Elements.globtrans(self::TensorElement{D}) where D = Shift(SVector{D,Float64}(self.index) - 1.0)


struct TensorDomain{D} <: Domain{TensorElement{D}, TensorReference{NTuple{D,SimplexReference{1}}}, D}
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
    factors = [insertaxis(basis1d[i,:]; left=i-1) for i in 1:D]
    outer = .*(factors...)
    reshape(outer, :)
end


end # module
