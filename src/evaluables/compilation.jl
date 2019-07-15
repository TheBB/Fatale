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


struct OptimizedEvaluable{I, K}
    funcs :: K

    function OptimizedEvaluable(func)
        sequence = linearize(func)
        callables = Tuple(codegen(stage.func) for stage in sequence)
        inds = Tuple(Tuple(stage.arginds) for stage in sequence)
        new{inds, typeof(callables)}(callables)
    end
end

struct OptimizedBlockEvaluable{I, D}
    indices :: I
    data :: D

    function OptimizedBlockEvaluable(block)
        indices = Tuple(OptimizedEvaluable(ind) for ind in block.indices)
        data = OptimizedEvaluable(block.data)
        new{typeof(indices), typeof(data)}(indices, data)
    end
end

struct OptimizedSparseEvaluable{K}
    blocks :: K

    function OptimizedSparseEvaluable(blocks)
        arg = Tuple(OptimizedBlockEvaluable(block) for block in blocks)
        new{typeof(arg)}(arg)
    end
end

"""
    optimize(::Evaluable)

Create an optimized and directly callable object from an evaluable.
"""
function optimize(self::Evaluable)
    blks = collect(Any, blocks(self))
    if length(blks) == 1 && _istrivial(blks[1]) && size(blks[1].data) == size(self)
        OptimizedEvaluable(self)
    else
        OptimizedSparseEvaluable(blks)
    end
end

function _istrivial(self)
    all(enumerate(self.indices)) do (i, ind)
        ind isa Constant && ind.value == 1:size(self.data, i)
    end
end


"""
    (::OptimizedEvaluable)(element, quadpt)

Evaluate the optimized evaluable in an evaluation point.
"""
@generated function (self::OptimizedEvaluable{Ind,K})(element, quadpt) where {Ind,K}
    nfuncs = length(K.parameters)
    syms = [gensym() for _ in 1:nfuncs]
    argsyms = [[syms[j] for j in tp] for tp in Ind]

    codes = Expr[]
    if quadpt == Nothing
        push!(codes, :(input = (element=element, point=(point=nothing, grad=nothing))))
    else
        N = length(quadpt)
        push!(codes, :(input = (element=element, point=(point=quadpt, grad=SMatrix{$N,$N,Float64}(I)))))
    end

    for (i, (functype, sym, args)) in enumerate(zip(K.parameters, syms, argsyms))
        code = :(self.funcs[$i](input, $(args...)))
        push!(codes, :($sym = $code))
    end

    quote
        $(codes...)
        $(syms[end])
    end
end
