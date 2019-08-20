# This file contains the most basic evaluables, mostly responsible for
# collecting evaluation arguments, element data, etc.


# Helper struct for allowing some evaluables to respond to different interfaces
# without multiple subtypes
struct Mimic
    eltype :: Union{Nothing, DataType}
    size :: Union{Nothing, Dims}
    ndims :: Union{Nothing, Int}

    function Mimic(T; size=nothing, eltype=nothing, ndims=nothing)
        T <: _Coords && @assert ndims isa Int
        T <: _Array && @assert size isa Dims
        T <: Union{_Coords,_Array} && @assert eltype isa DataType
        new(eltype, size, ndims)
    end
end

# Helper supertype for Mimics. Subtypes should have a field called
# mimic of type Mimic.
abstract type ShapeShifter{T} <: Evaluable{T} end

size(self::ShapeShifter{_Array}) = self.mimic.size
ndims(self::ShapeShifter{_Coords}) = self.mimic.ndims
eltype(self::ShapeShifter{<:Union{_Array,_Coords}}) = self.mimic.eltype


"""
    LocalPoint

Return the local quadrature point.
"""
struct LocalPoint <: ShapeShifter{_Array}
    mimic :: Mimic
    LocalPoint(n::Int, t::Type) = new(Mimic(_Array; size=(n,), eltype=t))
end

LocalPoint(n) = LocalPoint(n, Float64)

codegen(self::LocalPoint) = CplRawArg{1}()


"""
    LocalGrad

Return the local gradient.
"""
struct LocalGrad <: ShapeShifter{_Array}
    mimic :: Mimic
    LocalGrad(n::Int, t::Type) = new(Mimic(_Array; size=(n,n), eltype=t))
end

LocalGrad(n) = LocalGrad(n, Float64)

codegen(self::LocalGrad) = CplRawArg{2}()


"""
    EvalArg{T}(name; kwargs...)

Returns the evaluation argument named `name`.
"""
struct EvalArg{T} <: ShapeShifter{T}
    name :: Symbol
    mimic :: Mimic
    EvalArg{T}(name; kwargs...) where T = new{T}(name, Mimic(T; kwargs...))
end

codegen(self::EvalArg) = CplEvalArg{self.name}()


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
codegen(self::ElementData) = CplElementData{self.name}()


"""
    ExtractCoords(coords[, stage])

Extract the derivative at level `stage` from `coords`. Here, stage
zero corresponds to the point, stage one corresponds to the
derivative, etc.
"""
struct ExtractCoords <: ArrayEvaluable
    arg :: CoordsEvaluable
    stage :: Int
end

ExtractCoords(arg::CoordsEvaluable) = ExtractCoords(arg, 0)

arguments(self::ExtractCoords) = Evaluable[self.arg]
size(self::ExtractCoords) = ntuple(_->ndims(self.arg), self.stage+1)
codegen(self::ExtractCoords) = CplGetIndex{self.stage+1}()


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

codegen(self::ApplyTrans) = CplApplyTrans()
