struct Stage
    func :: Evaluable
    index :: Int
    arginds :: Vector{Int}
end

function linearize(self::Evaluable)
    data = OrderedDict{Evaluable, Stage}()
    _linearize!(data, self)
    collect(values(data))
end

function _linearize!(data, self::Evaluable)
    haskey(data, self) && return data[self].index
    arginds = Int[_linearize!(data, arg) for arg in arguments(self)]
    (data[self] = Stage(self, length(data) + 1, arginds)).index
end


abstract type AbstractOptimizedEvaluable{S,T} <: ArrayEvaluable end

Base.eltype(::AbstractOptimizedEvaluable{S,T}) where {S,T} = T
Base.size(::AbstractOptimizedEvaluable{S}) where S = S

optimize(self::AbstractOptimizedEvaluable) = self


# The most fundamental form of optimized evaluable.
# I: Tuple of integer tuples, I[k] is the results which form the
# inputs to evaluable number k
# K: Tuple of evaluable types
struct OptimizedEvaluable{I, K, S, T} <: AbstractOptimizedEvaluable{S,T}
    funcs :: K

    function OptimizedEvaluable(func)
        sequence = linearize(func)
        callables = Tuple(codegen(stage.func) for stage in sequence)
        inds = Tuple(Tuple(stage.arginds) for stage in sequence)
        new{inds, typeof(callables), size(func), eltype(func)}(callables)
    end
end

codegen(self::OptimizedEvaluable) = self


# A block is a dense array (an OptimizedEvaluable) together with a
# tuple of index evaluables, describing the variable placement of the
# dense array in a sparse superstructure.
struct OptimizedBlockEvaluable{I, D}
    indices :: I
    data :: D

    function OptimizedBlockEvaluable(block)
        indices = Tuple(OptimizedEvaluable(ind) for ind in block.indices)
        data = OptimizedEvaluable(block.data)
        new{typeof(indices), typeof(data)}(indices, data)
    end
end

OptimizedBlockEvaluable(block::OptimizedBlockEvaluable) = block

Base.eltype(self::OptimizedBlockEvaluable) = eltype(self.data)
Base.length(self::OptimizedBlockEvaluable) = length(self.data)
Base.ndims(self::OptimizedBlockEvaluable) = ndims(self.data)
Base.size(self::OptimizedBlockEvaluable) = size(self.data)


# An optimized sparse evaluable is a collection of one or more blocks.
struct OptimizedSparseEvaluable{K, S, T} <: AbstractOptimizedEvaluable{S,T}
    blocks :: K

    function OptimizedSparseEvaluable(blocks, func)
        arg = Tuple(OptimizedBlockEvaluable(block) for block in blocks)
        new{typeof(arg), size(func), eltype(func)}(arg)
    end
end

blocks(self::OptimizedSparseEvaluable) = self.blocks
SparseArrays.nnz(self::OptimizedSparseEvaluable) = sum(length, self.blocks)


"""
    optimize(::Evaluable)

Create an optimized and directly callable object from an evaluable.
"""
function optimize(self::ArrayEvaluable)
    blks = collect(Any, blocks(self))
    if length(blks) == 1 && _istrivial(blks[1]) && size(blks[1].data) == size(self)
        OptimizedEvaluable(self)
    else
        OptimizedSparseEvaluable(blks, self)
    end
end

_istrivial(self::OptimizedBlockEvaluable) = false
function _istrivial(self)
    all(enumerate(self.indices)) do (i, ind)
        ind isa Constant && ind.value == 1:size(self.data, i)
    end
end


@inline function (self::OptimizedEvaluable)(element, quadpt::Nothing)
    self((element=element, point=(point=nothing, grad=nothing)))
end

@inline function (self::OptimizedEvaluable)(element, quadpt::SVector{N,T}) where {N,T}
    grad = SMatrix{N,N,T}(I)
    self((element=element, point=(point=quadpt, grad=grad)))
end

"""
    (::OptimizedEvaluable)(evalargs)

Evaluate the optimized evaluable in an evaluation point. The argument
should be a named tuple of evaluation arguments, containing at least
*element* and *point*.
"""
@generated function (self::OptimizedEvaluable{Ind,K})(evalargs) where {Ind,K}
    nfuncs = length(K.parameters)
    syms = [gensym() for _ in 1:nfuncs]
    argsyms = [[syms[j] for j in tp] for tp in Ind]

    codes = Expr[]
    for (i, (functype, sym, args)) in enumerate(zip(K.parameters, syms, argsyms))
        code = :(self.funcs[$i](evalargs, $(args...)))
        push!(codes, :($sym = $code))
    end

    quote
        $(codes...)
        $(syms[end])
    end
end
