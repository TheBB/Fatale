# ==============================================================================
# Compilation blocks

# Supertype of all compilation blocks
abstract type CplBlock end

# Supertype of compilation blocks that are 'raw'.
# These receive the evaluation args as the first parameter, and the remaining
# arguments as targeted evaluation sequences.
abstract type RawCplBlock end

struct CplEvalArgs <: RawCplBlock end
@inline (::CplEvalArgs)(arg) = arg

struct CplFuncall{F,A} <: CplBlock end
@generated (::CplFuncall{F,A})(arg) where {F,A} = quote
    @_inline_meta
    $F(arg, :($A))
end

struct CplApplyTrans <: CplBlock end
@inline (::CplApplyTrans)(trans, coords) = apply(trans, coords)

struct CplConstant{T} <: CplBlock
    val :: T
end
@inline (self::CplConstant)() = self.val

struct CplGetIndex end
@inline (::CplGetIndex)(arg, indices...) = @inbounds arg[indices...]

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

struct CplMonomials{D,P,T}
    val :: T
    CplMonomials(degree, padding, eltype, size) = let val = @MArray zeros(eltype, size...)
        new{degree, padding, typeof(val)}(val)
    end
end
@generated function (self::CplMonomials{D,P})(arg) where {D,P}
    colons = [Colon() for _ in 1:ndims(arg)]
    codes = [
        :(self.val[$(colons...), $(P+i+1)] .= self.val[$(colons...), $(P+i)] .* arg)
        for i in 1:D
    ]

    quote
        @inbounds begin
            self.val[$(colons...), 1:$P] .= $(zero(eltype(arg)))
            self.val[$(colons...), $(P+1)] .= $(one(eltype(arg)))
            $(codes...)
        end
        SArray(self.val)
    end
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

    # We'd like to just call the StaticArrays implementation, but it
    # can cause allocations
    sums = Expr[]
    for i in Base.product((1:k for k in tempsize)...)
        ix = collect(i)
        exprs = Expr[]
        for px in Base.product((1:size(arg, d) for d in D)...)
            ix[D] = collect(px)
            push!(exprs, :(arg[$(indexer[ix...])]))
        end
        push!(sums, :(+($(exprs...))))
    end

    :(@inbounds SArray{Tuple{$(S...)}}($(sums...)))
end

struct CplContract{I,Ti,T}
    val :: T
    CplContract{I,Ti}(val::T) where {I,Ti,T} = new{I,Ti,T}(val)
end
@generated function (self::CplContract{I,Ti,T})(args...) where {I,Ti,T}
    dims = _sizedict(args, I)
    dim_order = Dict(axis => num for (num, axis) in enumerate(keys(dims)))

    codes = Expr[]
    for indices in product((1:n for n in values(dims))...)
        inputs = [
            :(args[$i][$((indices[dim_order[ax]] for ax in ind)...)])
            for (i, ind) in enumerate(I)
        ]
        product = :(*($(inputs...)))
        target = :(self.val[$((indices[dim_order[ax]] for ax in Ti)...)])
        push!(codes, :($target += $product))
    end

    quote
        @inbounds begin
            self.val .= $(zero(eltype(T)))
            $(codes...)
        end
        SArray(self.val)
    end
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

Base.length(self::Type{<:EvalSeq{I}}) where I = length(I)
Base.length(self::EvalSeq{I}) where I = length(I)


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
        if K.parameters[tgt] <: RawCplBlock
            return [:evalargs, (:(TargetedEvalSeq(self, Val($i))) for i in I[tgt])...]
        else
            return [syms[dep] for dep in I[tgt]]
        end
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
    functypes[tgt] <: RawCplBlock && return
    for dep in I[tgt]
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

Base.eltype(::AbstractOptimizedEvaluable{T}) where T = T
Base.ndims(::AbstractOptimizedEvaluable{T,N}) where {T,N} = N
Base.size(::AbstractOptimizedEvaluable{T,N,S}) where {T,N,S} = S

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

Base.eltype(self::OptimizedBlockEvaluable) = eltype(self.data)
Base.length(self::OptimizedBlockEvaluable) = length(self.data)
Base.ndims(self::OptimizedBlockEvaluable) = ndims(self.data)
Base.size(self::OptimizedBlockEvaluable) = size(self.data)


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
SparseArrays.nnz(self::OptimizedSparseEvaluable) = sum(length, self.blocks)


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
