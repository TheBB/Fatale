module Evaluables

import Base: @_inline_meta
import Base.Broadcast: broadcast_shape
import Base.Iterators: isdone, Stateful, flatten, product
using DataStructures
using LinearAlgebra
using StaticArrays
using UnsafeArrays

using ..Elements
using ..Transforms

export evalorder
export optimize
export local_point, local_grad, global_point, global_grad, grad, insertaxis
export Contract, Constant, Inv, Monomials, Product, Sum, Zeros


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


# Default implementations for array interface
Base.eltype(self::Evaluable{_Array}) = reduce(promote_type, map(eltype, arguments(self)))
Base.size(self::Evaluable{_Array}, i) = size(self)[i]
Base.ndims(self::Evaluable{_Array}) = length(size(self))
Base.length(self::Evaluable{_Array}) = prod(size(self))

# Evaluables of type Coords should also implement ndims, but it's not known in all cases
Base.ndims(::Evaluable{_Coords}) = "?"


include("evaluables/definitions.jl")
include("evaluables/utility.jl")
include("evaluables/gradients.jl")
include("evaluables/compilation.jl")


Base.show(io::IO, self::Evaluable) = print(io, string(typeof(self).name.name), typerepr(self))

typerepr(self::Evaluable{_Array}) = string("<", join(size(self), ","), ">")
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
