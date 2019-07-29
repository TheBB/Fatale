# This file contains the most basic evaluables, mostly responsible for
# collecting evaluation arguments, element data, etc.


"""
    EvalArgs()

A special Evaluable that just returns the evaluation arguments in raw
form. This is typically the only evaluable that receives undeclared
arguments.
"""
struct EvalArgs <: Evaluable{_Any} end

codegen(::EvalArgs) = __EvalArgs()
struct __EvalArgs end
@inline (::__EvalArgs)(arg) = arg

# This makes EvalArgs 'special', passing it the evaluation arguments
# directly. Other evaluables that need evaluation arguments should
# depend on EvalArgs instead of replicating this behaviour.
pass_evalargs(::Type{__EvalArgs}) = true


"""
    Funcall

A special Evaluable that represents a function call of the form

    functionname(argument, parameter)

Here, the function name is a symbol, the argument is another Evaluable
and the parameter is any object of bits type.

A funcall may represent any return type. Use one of the constructors:

    Funcall(_Array, functionname, argument, parameter, eltype, size)
    Funcall(_Coords, functionname, argument, parameter, eltype, (ndims,))
    Funcall(_Element, functionname, argument, parameter)
    Funcall(_Transform, functionname, argument, parameter)

E.g. for field access or element data use

    Funcall(..., :getfield, argument, :fieldname, ...)
    Funcall(..., :elementdata, argument, :dataname, ...)
"""
struct Funcall{T} <: Evaluable{T}
    funcname :: Symbol
    argument :: Evaluable
    parameter

    # This is a hack so that one struct can represent many different output types
    eltype :: Union{Nothing, DataType}
    size :: Union{Nothing, Dims}
    ndims :: Union{Nothing, Int}

    function Funcall{T}(func, arg, param; size=nothing, eltype=nothing, ndims=nothing) where T
        T <: _Coords && @assert ndims isa Int
        T <: _Array && @assert size isa Dims
        T <: Union{_Coords,_Array} && @assert eltype isa DataType
        @assert isbits(param) || param isa Symbol
        new{T}(func, arg, param, eltype, size, ndims)
    end
end

# There are special outer constructors for each result type
Funcall(::Type{_Coords}, func, arg, param, eltype, (n,)) = Funcall{_Coords}(func, arg, param; eltype=eltype, ndims=n)
Funcall(::Type{_Array}, func, arg, param, eltype, size) = Funcall{_Array}(func, arg, param; eltype=eltype, size=size)
Funcall(T::Type{<:Result}, func, arg, param) = Funcall{T}(func, arg, param)

arguments(self::Funcall) = Evaluable[self.argument]
Base.size(self::Funcall{_Array}) = self.size
Base.ndims(self::Funcall{_Coords}) = self.ndims
Base.eltype(self::Funcall{<:Union{_Array,_Coords}}) = self.eltype

codegen(self::Funcall) = __Funcall{self.funcname, self.parameter}()
struct __Funcall{F,A} end
@generated function (::__Funcall{F,A})(arg) where {F,A}
    quote
        @_inline_meta
        $F(arg, :($A))
    end
end


"""
    ApplyTrans(trans, coords)

Apply the transformation *trans* to *coords*.
"""
struct ApplyTrans <: CoordsEvaluable
    transform :: Evaluable{_Transform}
    coords :: CoordsEvaluable
end

Base.eltype(self::ApplyTrans) = eltype(self.coords)
Base.ndims(self::ApplyTrans) = ndims(self.coords)
arguments(self::ApplyTrans) = Evaluable[self.transform, self.coords]

codegen(self::ApplyTrans) = __ApplyTrans()
struct __ApplyTrans end
@inline (::__ApplyTrans)(trans, point) = apply(trans, point)
