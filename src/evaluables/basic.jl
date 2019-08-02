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

Base.size(self::ShapeShifter{_Array}) = self.mimic.size
Base.ndims(self::ShapeShifter{_Coords}) = self.mimic.ndims
Base.eltype(self::ShapeShifter{<:Union{_Array,_Coords}}) = self.mimic.eltype


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
Base.size(self::ExtractCoords) = ntuple(_->ndims(self.arg), self.stage+1)
codegen(self::ExtractCoords) = CplGetIndex{self.stage+1}()


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

codegen(self::ApplyTrans) = CplApplyTrans()
