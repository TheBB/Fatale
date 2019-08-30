# ==============================================================================
# Linearization

# Internal structure for a single stage in a linearized evaluation sequence
struct Stage
    func :: Evaluable
    index :: Int
    arginds :: Vector{Int}
end

"""
    linearize(::Evaluable) :: Vector{Stage}

Convert an evaluable tree into a linear sequence of evaluables, such
that any evaluable depends only on results of previous evaluables.
"""
function linearize(self::Evaluable)
    data = OrderedDict{Evaluable, Stage}()
    _linearize!(data, self)
    collect(values(data))
end

# Internal worker for linearize
function _linearize!(data, self::Evaluable)
    haskey(data, self) && return data[self].index
    arginds = Int[_linearize!(data, arg) for arg in arguments(self)]
    (data[self] = Stage(self, length(data) + 1, arginds)).index
end


# ==============================================================================
# Evaluation sequences

# I: Tuple of integer tuples, I[k] are all functions indices whose
# outputs form the inputs of the function at index k.
struct EvalSeq{I,K}
    funcs :: K

    function EvalSeq(func::Evaluable)
        sequence = linearize(func)
        callables = Tuple(codegen(stage.func) for stage in sequence)
        inds = Tuple(Tuple(stage.arginds) for stage in sequence)
        new{inds, typeof(callables)}(callables)
    end
end

length(self::Type{<:EvalSeq{I}}) where I = length(I)
length(self::EvalSeq{I}) where I = length(I)


"""
    (::EvalSeq)([index::Val{k},] evalargs::NamedTuple)

Evaluate an evaluation sequence at some collection of
arguments. Returns the result of the function at index *k*, or the
result of the full evaluation sequence if not given.
"""
@generated function (self::EvalSeq{I,K})(::Val{N}, point, locgrad, evalargs::NamedTuple) where {N,I,K}
    seq = _sequence(I, N, K.parameters)
    syms = [gensym() for _ in 1:length(self)]

    argexprs = map(enumerate(seq)) do (i, tgt)
        iraw = 0
        special_args = map(Cpl.argspec(K.parameters[tgt])) do spec
            spec in (:point, :locgrad, :evalargs) && return spec
            spec == :rawarg && return :(TargetedEvalSeq(self, Val($(I[tgt][iraw+=1]))))
            @assert false       # should be unreachable
        end
        natural_args = Union{Expr,Symbol}[syms[dep] for dep in I[tgt]]
        return [special_args..., drop(natural_args, iraw)...]
    end

    codes = map(zip(seq, syms[seq], argexprs)) do (i, sym, args)
        :($sym = self.funcs[$i]($(args...)))
    end

    quote
        $(codes...)
        $(syms[N])
    end
end

@generated (self::EvalSeq)(evalargs::NamedTuple) = quote
    @_inline_meta
    self(Val($(length(self))), evalargs)
end

function _sequence(I, N, functypes)
    seq = Set{Int}()
    _sequence!(seq, I, N, functypes)
    sort(collect(seq))
end

function _sequence!(ret, I, tgt, functypes)
    push!(ret, tgt)
    for (idx, dep) in enumerate(I[tgt])
        idx <= count(==(:rawarg), Cpl.argspec(functypes[tgt])) && continue
        _sequence!(ret, I, dep, functypes)
    end
end


# Helper struct for bundling together an evaluation sequence and a target function index
struct TargetedEvalSeq{S<:EvalSeq, I<:Val}
    sequence :: S
    target :: I
end

TargetedEvalSeq(seq::EvalSeq) = TargetedEvalSeq(seq, Val(length(seq)))
TargetedEvalSeq(func::Evaluable) = TargetedEvalSeq(EvalSeq(func))

@inline (self::TargetedEvalSeq)(point, locgrad, evalargs::NamedTuple) =
    self.sequence(self.target, point, locgrad, evalargs)

pass_evalargs(::Type{TargetedEvalSeq}) = true


# ==============================================================================
# Optimized evaluables

# Generally, an evaluation sequence bundled with size and eltype information
# so that they can themselves act as evaluables.

"""
    AbstractOptimizedEvaluable{T,N,S}

A compiled evaluable of element type T, number of dimensions N and
size S.
"""
abstract type AbstractOptimizedEvaluable{T,N,S} <: ArrayEvaluable end

eltype(::AbstractOptimizedEvaluable{T}) where T = T
ndims(::AbstractOptimizedEvaluable{T,N}) where {T,N} = N
size(::AbstractOptimizedEvaluable{T,N,S}) where {T,N,S} = S

optimize(self::AbstractOptimizedEvaluable) = self


"""
    OptimizedEvaluable <: AbstractOptimizedEvaluable

The most fundamental form of optimized evaluable.
"""
struct OptimizedEvaluable{T,N,S,F<:TargetedEvalSeq} <: AbstractOptimizedEvaluable{T,N,S}
    sequence :: F

    function OptimizedEvaluable(func::ArrayEvaluable)
        seq = TargetedEvalSeq(func)
        new{eltype(func), ndims(func), size(func), typeof(seq)}(seq)
    end
end

codegen(self::OptimizedEvaluable) = self.sequence

"""
    (func::OptimizedEvaluable)(evalargs::NamedTuple)
    (func::OptimizedEvaluable)(element, quadpt)

Evaluate an optimized evaluable with a collection of input arguments.
The second call is equivalent to a minimal argument tuple with
*element* and *coords* (the latter of which may be Nothing).
"""
@inline (self::OptimizedEvaluable)(element::Maybe{AbstractElement}, quadpt::Nothing) =
    self(nothing, nothing, (element=element,))

@inline function (self::OptimizedEvaluable)(element::AbstractElement, quadpt::SVector)
    trans = elementdata(element, Val{:loctrans}())
    point, locgrad = apply(trans, (point=quadpt, grad=nothing))
    self(point, locgrad, (element=element,))
end

@inline (self::OptimizedEvaluable)(point::Maybe{SVector}, locgrad::Maybe{SMatrix}, evalargs::NamedTuple) =
    self.sequence(point, locgrad, evalargs)


# A block is a dense array (an OptimizedEvaluable) together with a
# tuple of index evaluables, describing the variable placement of the
# dense array in a sparse superstructure.
struct OptimizedBlockEvaluable{N, I<:VarTuple{OptimizedEvaluable}, D<:OptimizedEvaluable}
    indices :: I
    data :: D

    function OptimizedBlockEvaluable(block)
        indices = Tuple(OptimizedEvaluable(ind) for ind in block.indices)
        data = OptimizedEvaluable(block.data)
        new{length(indices), typeof(indices), typeof(data)}(indices, data)
    end
end

OptimizedBlockEvaluable(block::OptimizedBlockEvaluable) = block

eltype(self::OptimizedBlockEvaluable) = eltype(self.data)
length(self::OptimizedBlockEvaluable) = length(self.data)
ndims(self::OptimizedBlockEvaluable) = ndims(self.data)
size(self::OptimizedBlockEvaluable) = size(self.data)


"""
    OptimizedSparseEvaluable <: AbstractOptimizedEvaluable

An optimized sparse evaluable is a collection of evaluation blocks,
each of type OptimizedBlockEvaluable.
"""
struct OptimizedSparseEvaluable{T,N,S,K} <: AbstractOptimizedEvaluable{T,N,S}
    blocks :: K

    function OptimizedSparseEvaluable(blocks, func)
        arg = Tuple(OptimizedBlockEvaluable(block) for block in blocks)
        new{eltype(func), ndims(func), size(func), typeof(arg)}(arg)
    end
end

blocks(self::OptimizedSparseEvaluable) = self.blocks
nnz(self::OptimizedSparseEvaluable) = sum(length, self.blocks)


"""
    optimize(::Evaluable)

Create an optimized and directly callable object from an evaluable.

If the evaluable can be determined to be trivial, this will return an OptimizedEvaluable.
Otherwise it will return an OptimizedSparseEvaluable.

A trivial evaluable consists of one block with OneTo-type indices.
"""
function optimize(self::ArrayEvaluable)
    blks = collect(Any, blocks(self))
    if length(blks) == 1 && _istrivial(blks[1]) && size(blks[1].data) == size(self)
        OptimizedEvaluable(self)
    else
        OptimizedSparseEvaluable(blks, self)
    end
end

# Detection of 'trivial' blocks
_istrivial(self::OptimizedBlockEvaluable) = false
_istrivial(self) = all(enumerate(self.indices)) do (i, ind)
    ind isa AbstractConstant && valueof(ind) == 1:size(self.data, i)
end
