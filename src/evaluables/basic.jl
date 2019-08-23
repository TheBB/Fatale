# This file contains the most basic evaluables, mostly responsible for
# collecting evaluation arguments, element data, etc.


# Helper struct for allowing some evaluables to respond to array interface
struct Mimic
    eltype :: Union{Nothing, DataType}
    size :: Union{Nothing, Dims}

    function Mimic(T; size=nothing, eltype=nothing)
        T <: _Array && @assert size isa Dims
        T <: _Array && @assert eltype isa Type
        new(eltype, size)
    end
end

# Helper supertype for Mimics. Subtypes should have a field called
# mimic of type Mimic.
abstract type ShapeShifter{T} <: Evaluable{T} end

size(self::ShapeShifter{_Array}) = self.mimic.size
eltype(self::ShapeShifter{_Array}) = self.mimic.eltype


"""
    LocalPoint

Return the local quadrature point.
"""
struct LocalPoint <: ShapeShifter{_Array}
    mimic :: Mimic
    LocalPoint(n::Int, t::Type) = new(Mimic(_Array; size=(n,), eltype=t))
end

LocalPoint(n) = LocalPoint(n, Float64)

codegen(self::LocalPoint) = Cpl.RawArg{1}()


"""
    LocalGrad

Return the local gradient.
"""
struct LocalGrad <: ShapeShifter{_Array}
    mimic :: Mimic
    LocalGrad(n::Int, t::Type) = new(Mimic(_Array; size=(n,n), eltype=t))
end

LocalGrad(n) = LocalGrad(n, Float64)

codegen(self::LocalGrad) = Cpl.RawArg{2}()


"""
    EvalArg{T}(name; kwargs...)

Returns the evaluation argument named `name`.
"""
struct EvalArg{T} <: ShapeShifter{T}
    name :: Symbol
    mimic :: Mimic
    EvalArg{T}(name; kwargs...) where T = new{T}(name, Mimic(T; kwargs...))
end

codegen(self::EvalArg) = Cpl.EvalArg{self.name}()


"""
    ElementData{T}(name; kwargs...)

Returns the element data named `name`.
"""
struct ElementData{T} <: ShapeShifter{T}
    name :: Symbol
    mimic :: Mimic
    ElementData{T}(name; kwargs...) where T = new{T}(name, Mimic(T; kwargs...))
end

arguments(self::ElementData) = Evaluable[EvalArg{_Element}(:element)]
codegen(self::ElementData) = Cpl.ElementData{self.name}()


"""
    ApplyTrans(trans, coords)

Apply the transformation *trans* to *coords*.
"""
struct ApplyTrans <: ArrayEvaluable
    transform :: Evaluable{_Transform}
    coords :: ArrayEvaluable
end

eltype(self::ApplyTrans) = eltype(self.coords)
size(self::ApplyTrans) = size(self.coords)
arguments(self::ApplyTrans) = Evaluable[self.transform, self.coords]

codegen(self::ApplyTrans) = Cpl.ApplyTrans()
