# ==============================================================================
# Compilation blocks

# True if evalargs should be passed
pass_evalargs(::Type{<:Any}) = false

# Tuple of argument indices that should be passed in raw form
raw_args(::Type{<:Any}) = ()

struct CplEvalArg{T} end
@generated (::CplEvalArg{T})(arg) where T = quote
    @_inline_meta
    arg.$T
end
pass_evalargs(::Type{<:CplEvalArg}) = true

struct CplElementData{T} end
@generated (::CplElementData{T})(element) where T = quote
    @_inline_meta
    elementdata(element, Val(:($T)))
end

struct CplApplyTrans end
@inline (::CplApplyTrans)(trans, coords) = apply(trans, coords)

struct CplConstant{T}
    val :: T
end
@inline (self::CplConstant)() = self.val

struct CplGetIndex{T} end
(::Type{CplGetIndex})() = CplGetIndex{Nothing}()
@inline (::CplGetIndex{Nothing})(arg, indices...) = @inbounds arg[indices...]
@inline (::CplGetIndex{T})(arg) where T = @inbounds arg[T]

struct CplInv end
@inline (::CplInv)(arg) = inv(arg)

struct CplNegate end
@inline (::CplNegate)(arg) = -arg

struct CplPower{P} end
@inline (::CplPower{P})(arg::Scalar) where P = Scalar(arg[] ^ P) # Remove when StaticArrays bug fixed
@inline (::CplPower{P})(arg) where P = arg .^ P

struct CplReciprocal end
@inline (::CplReciprocal)(arg::Scalar) = Scalar(one(eltype(arg)) / arg[]) # Remove when StaticArrays bug fixed
@inline (::CplReciprocal)(arg) = one(eltype(arg)) ./ arg

struct CplReshape{S} end
@inline (::CplReshape{S})(arg) where S = SArray{Tuple{S...}}(arg)

struct CplSqrt end
@inline (::CplSqrt)(arg::Scalar) = Scalar(sqrt(arg[])) # Remove when StaticArrays bug fixed
@inline (::CplSqrt)(arg) = sqrt.(arg)

struct CplCommArith{F} end
@generated (::CplCommArith{F})(args...) where F = quote
    @_inline_meta
    $F(args...)
end

struct CplElementIntegral{T} end
@inline function (self::CplElementIntegral{T})(args, sub, loctrans, quadrule) where T
    temp = zero(T)
    (pts, wts) = quadrule
    for (pt, wt) in zip(pts, wts)
        coords = apply(loctrans, (point=pt, grad=nothing))
        temp = temp .+ sub((args..., coords=coords)) .* wt
    end
    temp
end
pass_evalargs(::Type{<:CplElementIntegral}) = true
raw_args(::Type{<:CplElementIntegral}) = (1,)

struct CplMonomials{D,P,T} end
@generated function (self::CplMonomials{D,P,T})(arg) where {D,P,T}
    exprs = [
        i <= P ? zero(eltype(arg)) : :(arg[$j] ^ $(i-P-1))
        for i in 1:(P+D+1)
        for j in CartesianIndices(size(arg))
    ]
    :(@inbounds $T($(exprs...)))
end

struct CplPermuteDims{I} end
@generated function (::CplPermuteDims{I})(arg) where I
    insize = size(arg)
    outsize = Tuple(size(arg, i) for i in I)

    lininds = LinearIndices(insize)
    indices = (lininds[(cind[i] for i in I)...] for cind in CartesianIndices(outsize))
    exprs = (:(arg[$i]) for i in indices)
    quote
        @_inline_meta
        SArray{Tuple{$(outsize...)}}($(exprs...))
    end
end

struct CplSum{D,S} end
@generated function (self::CplSum{D,S})(arg) where {D,S}
    D = collect(D)
    tempsize = Tuple(i in D ? 1 : k for (i, k) in enumerate(size(arg)))
    indexer = LinearIndices(size(arg))

    out_indices = product((1:k for k in tempsize)...)
    sums = map(out_indices) do out_ind
        in_ind = collect(out_ind)
        sum_indices = product((1:size(arg, d) for d in D)...)
        exprs = map(sum_indices) do sum_ind
            in_ind[D] = collect(sum_ind)
            return :(arg[$(indexer[in_ind...])])
        end
        return :(+($(exprs...)))
    end

    :(@inbounds SArray{Tuple{$(S...)}}($(sums...)))
end

struct CplContract{I,Ti,T} end
@generated function (self::CplContract{I,Ti,T})(args...) where {I,Ti,T}
    dims = _sizedict(args, I)

    # dim_order maps an axis label to an arbitrary one-based index
    dim_order = Dict(axis => num for (num, axis) in enumerate(keys(dims)))

    # getind(indmap, i) unpacks the indices in i corresponding to the index labels in indmap
    getind = (indmap, i) -> (i[dim_order[ax]] for ax in indmap)

    sums = Vector{Expr}(undef, length(T))
    for indices in product((1:n for n in values(dims))...)
        in_factors = map(enumerate(I)) do (i, ind)
            index = LinearIndices(size(args[i]))[getind(ind, indices)...]
            return :(args[$i][$index])
        end
        in_prod = :(*($(in_factors...)))
        out_index = LinearIndices(size(T))[getind(Ti, indices)...]
        sums[out_index] = isassigned(sums, out_index) ? :($(sums[out_index]) + $in_prod) : in_prod
    end

    :(@inbounds $T($(sums...)))
end


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
@generated function (self::EvalSeq{I,K})(::Val{N}, evalargs::NamedTuple) where {N,I,K}
    seq = _sequence(I, N, K.parameters)
    syms = [gensym() for _ in 1:length(self)]

    argexprs = map(enumerate(seq)) do (i, tgt)
        args = Union{Expr,Symbol}[syms[dep] for dep in I[tgt]]

        blocktype = K.parameters[tgt]
        for idx in raw_args(blocktype)
            args[idx] = :(TargetedEvalSeq(self, Val($(I[tgt][idx]))))
        end

        pass_evalargs(blocktype) && pushfirst!(args, :evalargs)
        return args
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
        idx in raw_args(functypes[tgt]) && continue
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

@inline (self::TargetedEvalSeq)(evalargs::NamedTuple) = self.sequence(self.target, evalargs)

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
@inline function (self::OptimizedEvaluable)(element, quadpt::Nothing)
    self((element=element, coords=(point=nothing, grad=nothing)))
end

@inline function (self::OptimizedEvaluable)(element, quadpt::SVector{N,T}) where {N,T}
    trans = elementdata(element, Val{:loctrans}())
    coords = apply(trans, (point=quadpt, grad=nothing))
    self((element=element, coords=coords))
end

@inline (self::OptimizedEvaluable)(evalargs::NamedTuple) = self.sequence(evalargs)


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
