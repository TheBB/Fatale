module Evaluables

using Base: @_inline_meta
using Base.Broadcast: broadcast_shape
using Base.Iterators: isdone, Stateful, flatten, product, countfrom
using DataStructures: OrderedDict
using StaticArrays: MArray, SArray, Scalar, SOneTo, SUnitRange, SVector
using ..Elements: elementdata
using ..Transforms: apply, isupdim, splittrf
using ..Utils: MemoizedMap

import Base: eltype, size, ndims, length, map, firstindex, lastindex, show
import Base: convert, broadcastable, BroadcastStyle, literal_pow, *, -, +
import Base: getindex, inv, reshape, adjoint, transpose, permutedims, sum
import Base.Broadcast: materialize, Broadcasted
import LinearAlgebra: dot, normalize, norm, norm_sqr
import SparseArrays: nnz

export evalorder, optimize, blocks
export local_point, local_grad, global_point, global_grad, grad, insertaxis, element_index, normal
export Contract, Constant, ElementIntegral, Inflate, Monomials, Zeros


abstract type Result end
struct _Coords <: Result end
struct _Array <: Result end
struct _Element <: Result end
struct _Transform <: Result end
struct _Any <: Result end


"""
    Evaluable{T}

An object that, when evaluated, produces a value of type T.
"""
abstract type Evaluable{T <: Result} end

"""
    arguments(::Evaluable)

Return a list of evaluables forming the arguments to the given
evaluable.
"""
arguments(::Evaluable) = Evaluable[]

"""
    map(f, ::Evaluable)

Call *f* for each evaluable in the tree.
"""
function map(f, self::Evaluable)
    f(self)
    for arg in arguments(self)
        map(f, arg)
    end
end

# Some type aliases that will be useful
const ArrayEvaluable = Evaluable{_Array}
const CoordsEvaluable = Evaluable{_Coords}
const VarTuple{K} = Tuple{Vararg{K}}

# Default implementations for array interface
eltype(self::ArrayEvaluable) = mapreduce(eltype, promote_type, arguments(self))
size(self::ArrayEvaluable, i) = size(self)[i]
ndims(self::ArrayEvaluable) = length(size(self))
length(self::ArrayEvaluable) = prod(size(self))

firstindex(self::ArrayEvaluable) = ntuple(_->1, ndims(self))
firstindex(::ArrayEvaluable, i) = 1
lastindex(self::ArrayEvaluable) = size(self)
lastindex(self::ArrayEvaluable, i) = size(self, i)


eltype(::CoordsEvaluable) = throw("not implemented")
ndims(::CoordsEvaluable) = throw("not implemented")


# Supertype for all evaluables with constant value
abstract type AbstractConstant <: ArrayEvaluable end
valueof(::AbstractConstant) = throw("not implemented")
codegen(self::AbstractConstant) = CplConstant(valueof(self))


"""
    blocks(::ArrayEvaluable)

Decompose an array evaluable into sparse blocks. Returns an iterator
of type (indices=(...), data=...), with evaluables producing
IJV-format array elements.
"""
blocks(self::ArrayEvaluable) = [(
    indices = Tuple(OneTo(s) for s in size(self)),
    data = self,
)]


include("evaluables/compilation.jl")

include("evaluables/basic.jl")
include("evaluables/constants.jl")
include("evaluables/definitions.jl")
include("evaluables/arithmetic.jl")
include("evaluables/contract.jl")

include("evaluables/utility.jl")
include("evaluables/gradients.jl")


show(io::IO, self::Evaluable) = print(io, string(typeof(self).name.name), typerepr(self))

typerepr(self::Evaluable) = ""
typerepr(self::ArrayEvaluable) = string("<", join(size(self), ","), ">")
typerepr(self::CoordsEvaluable) = let n = ndims(self); "(point=<$n>, grad=<$n,$n>)" end
typerepr(self::Evaluable{_Element}) = "(Element)"
typerepr(self::Evaluable{_Transform}) = "(Transform)"


"""
    evalorder(::Evaluable)

Return a string with a human-readable representation of the evaluation
sequence of an evaluable.
"""
function evalorder(self::Evaluable)
    sequence = linearize(self)
    join((string("%", stage.index, " = ", repr(stage.func),
                 "(", join((string("%", argind) for argind in stage.arginds), ", "), ")")
          for stage in sequence),
         "\n")
end


end # module
