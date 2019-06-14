module Evaluables

import Base: @_inline_meta
import Base.Iterators: isdone, Stateful
using DataStructures
using StaticArrays
using LinearAlgebra

using ..Elements

export evalorder
export optimize
export localpoint, localgrad, globalpoint, globalgrad


"""
    Evaluable{T}

An object that, when evaluated, produces a value of type T.
"""
abstract type Evaluable{T} end

restype(::Evaluable{T}) where T = T
restype(::Type{<:Evaluable{T}}) where T = T
arguments(::Evaluable) = Evaluable[]

# The default hash and equals behaviour on evaluables is based
# strictly on the types and arguments involved. This should be
# sufficient for the vast majority of evaluables, but some
# (e.g. constants) may override it.
Base.hash(self::Evaluable, x::UInt64) = hash(typeof(self), hash(arguments(self), x))
Base.:(==)(l::Evaluable, r::Evaluable) = typeof(l) == typeof(r) && arguments(l) == arguments(r)

# Code generation for most evaluables involves calling the evaluable
# object directly.
codegen(::Type{<:Evaluable}, self, args...) = :($self($(args...)))


# Most of our evaluables produce arrays, but we postpone the decision
# on exactly which type of array until the compilation stage. Before
# that, we use the abstract `StaticArray` type.
const ArrayEvaluable{S <: Tuple, T, N} = Evaluable{StaticArray{S, T, N}}
const ScalarEvaluable{T} = ArrayEvaluable{Tuple{}, T, 0}
const VectorEvaluable{N, T} = ArrayEvaluable{Tuple{N}, T, 1}
const MatrixEvaluable{M, N, T} = ArrayEvaluable{Tuple{N, M}, T, 2}


include("evaluables/definitions.jl")
include("evaluables/utility.jl")
include("evaluables/compilation.jl")


Base.show(io::IO, self::Evaluable{T}) where T = print(io, string(typeof(self).name.name), _typerepr(T))

_typerepr(::Type{T}) where T <: StaticArray = string("<", join(size(T), ","), ">")
_typerepr(::Type{T}) where T <: Tuple = string("(", join((_typerepr(param) for param in T.parameters), ", "), ")")

function _typerepr(::Type{T}) where T <: NamedTuple
    names = T.parameters[1]
    values = T.parameters[2].parameters
    entries = join((string(name, "=", _typerepr(value)) for (name, value) in zip(names, values)), ", ")
    string("{", entries, "}")
end


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
