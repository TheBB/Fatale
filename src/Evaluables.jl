module Evaluables

import Base: @_inline_meta
import Base.Broadcast: broadcast_shape
import Base.Iterators: isdone, Stateful, flatten, product
using DataStructures
using LinearAlgebra
using StaticArrays
using UnsafeArrays
using SparseArrays

using ..Elements
using ..Transforms

export evalorder
export optimize, blocks
export local_point, local_grad, global_point, global_grad, grad, insertaxis, element_index, normal
export Contract, Constant, Inflate, Monomials, Zeros


abstract type Result end
struct _Coords <: Result end
struct _Array <: Result end
struct _Element <: Result end
struct _Transform <: Result end


"""
    Evaluable{T}

An object that, when evaluated, produces a value of type T.
"""
abstract type Evaluable{T <: Result} end

arguments(::Evaluable) = Evaluable[]

# Some type aliases that will be useful
const ArrayEvaluable = Evaluable{_Array}
const VarTuple{K} = Tuple{Vararg{K}}

# Default implementations for array interface
Base.eltype(self::ArrayEvaluable) = mapreduce(eltype, promote_type, arguments(self))
Base.size(self::ArrayEvaluable, i) = size(self)[i]
Base.ndims(self::ArrayEvaluable) = length(size(self))
Base.length(self::ArrayEvaluable) = prod(size(self))

Base.firstindex(self::ArrayEvaluable) = ntuple(_->1, ndims(self))
Base.firstindex(self::ArrayEvaluable, i) = 1
Base.lastindex(self::ArrayEvaluable) = size(self)
Base.lastindex(self::ArrayEvaluable, i) = size(self, i)

# Evaluables of type Coords should also implement ndims, but it's not known in all cases
Base.ndims(::Evaluable{_Coords}) = "?"


"""
    blocks(::ArrayEvaluable)

Decompose an array evaluable into sparse blocks. Returns an iterator
of type (indices=(...), data=...), with evaluables producing
IJV-format array elements.
"""
blocks(self::ArrayEvaluable) = [(
    indices = Tuple(Constant(1:s) for s in size(self)),
    data = self,
)]


include("evaluables/definitions.jl")
include("evaluables/utility.jl")
include("evaluables/gradients.jl")
include("evaluables/compilation.jl")


Base.show(io::IO, self::Evaluable) = print(io, string(typeof(self).name.name), typerepr(self))

typerepr(self::ArrayEvaluable) = string("<", join(size(self), ","), ">")
typerepr(self::Evaluable{_Coords}) = let n = ndims(self); "(point=<$n>, grad=<$n,$n>)" end
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
