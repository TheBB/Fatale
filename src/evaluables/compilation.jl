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


struct OptimizedEvaluable{T, I, K} <: Evaluable{T}
    funcs :: K
    OptimizedEvaluable{T,I}(funcs::K) where {T,I,K} = new{T,I,K}(funcs)
end

"""
    optimize(::Evaluable)

Create an optimized and directly callable object from an evaluable.
"""
function optimize(self::Evaluable{T}) where T
    sequence = linearize(self)
    callables = Tuple(codegen(stage.func) for stage in sequence)
    inds = Tuple{(Tuple{stage.arginds...} for stage in sequence)...}
    OptimizedEvaluable{T, inds}(callables)
end

"""
    (::OptimizedEvaluable)(element, quadpt)

Evaluate the optimized evaluable in an evaluation point.
"""
@generated function (self::OptimizedEvaluable{T,Ind,K})(element, quadpt) where {T,Ind,K}
    nfuncs = length(K.parameters)
    syms = [gensym() for _ in 1:nfuncs]
    argsyms = [[syms[j] for j in tp.parameters] for tp in Ind.parameters]

    codes = Expr[]
    if quadpt == Nothing
        push!(codes, :(input = (element=element, point=(point=nothing, grad=nothing))))
    else
        N = length(quadpt)
        push!(codes, :(input = (element=element, point=(point=quadpt, grad=SMatrix{$N,$N,Float64}(I)))))
    end

    for (i, (functype, sym, args)) in enumerate(zip(K.parameters, syms, argsyms))
        code = :(self.funcs[$i](input, $(args...)))
        # code = codegen(functype, :(self.funcs[$i]), :input, args...)
        push!(codes, :($sym = $code))
    end

    quote
        $(codes...)
        $(syms[end])
    end
end
