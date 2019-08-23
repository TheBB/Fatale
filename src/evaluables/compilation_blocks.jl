module CompilationBlocks

using Base: @_inline_meta
using Base.Iterators: product
using ForwardDiff: jacobian
using StaticArrays: SArray, Scalar

using ...Elements: elementdata
using ...Transforms: apply, isupdim
import ..Evaluables             # for access to some helpers


abstract type CompilationBlock end

"""
    pass_evalargs(::Type{<:CompilationBlock})

Return true if the compilation block should have raw access to the
evaluation arguments, in addition to just its own arguments.
"""
pass_evalargs(::Type{<:CompilationBlock}) = false

"""
    raw_args(::Type{<:CompilationBlock})

Return the indices of the arguments that should be passed in 'raw'
form (as evaluables) as opposed to just their outputs.
"""
raw_args(::Type{<:CompilationBlock}) = ()


struct RawArg{N} <: CompilationBlock end
@inline (::RawArg{N})(args...) where N = args[N]
pass_evalargs(::Type{<:RawArg}) = true

struct EvalArg{T} <: CompilationBlock end
@generated (::EvalArg{T})(_, _, evalargs) where T = quote
    @_inline_meta
    evalargs.$T
end
pass_evalargs(::Type{<:EvalArg}) = true

struct ElementData{T} <: CompilationBlock end
@generated (::ElementData{T})(element) where T = quote
    @_inline_meta
    elementdata(element, Val(:($T)))
end

struct ApplyTrans <: CompilationBlock end
@inline function (::ApplyTrans)(trans, coords)
    @assert !isupdim(trans)
    apply(trans, coords)
end

struct Constant{T} <: CompilationBlock
    val :: T
end
@inline (self::Constant)() = self.val

struct GetIndex{T} <: CompilationBlock end
(::Type{GetIndex})() = GetIndex{Nothing}()
@inline (::GetIndex{Nothing})(arg, indices...) = @inbounds arg[indices...]
@inline (::GetIndex{T})(arg) where T = @inbounds arg[T]

struct Gradient{S} <: CompilationBlock end
@inline function (::Gradient{S})(point, locgrad, evalargs, arg) where S
    subgrad = transpose(jacobian(x -> arg(x, locgrad, evalargs), point))
    return SArray{Tuple{S...}}(subgrad)
end
pass_evalargs(::Type{<:Gradient}) = true
raw_args(::Type{<:Gradient}) = (1,)

struct Inv <: CompilationBlock end
@inline (::Inv)(arg) = inv(arg)

struct Negate <: CompilationBlock end
@inline (::Negate)(arg) = -arg

struct Power{P} <: CompilationBlock end
@inline (::Power{P})(arg::Scalar) where P = Scalar(arg[] ^ P) # Remove when StaticArrays bug fixed
@inline (::Power{P})(arg) where P = arg .^ P

struct Reciprocal <: CompilationBlock end
@inline (::Reciprocal)(arg::Scalar) = Scalar(one(eltype(arg)) / arg[]) # Remove when StaticArrays bug fixed
@inline (::Reciprocal)(arg) = one(eltype(arg)) ./ arg

struct Reshape{S} <: CompilationBlock end
@inline (::Reshape{S})(arg) where S = SArray{Tuple{S...}}(arg)

struct Sqrt <: CompilationBlock end
@inline (::Sqrt)(arg::Scalar) = Scalar(sqrt(arg[])) # Remove when StaticArrays bug fixed
@inline (::Sqrt)(arg) = sqrt.(arg)

struct CommArith{F} <: CompilationBlock end
@generated (::CommArith{F})(args...) where F = quote
    @_inline_meta
    $F(args...)
end

struct ElementIntegral{T} <: CompilationBlock end
@inline function (self::ElementIntegral{T})(_, _, args, sub, loctrans, quadrule) where T
    temp = zero(T)
    (pts, wts) = quadrule
    for (pt, wt) in zip(pts, wts)
        point, locgrad = apply(loctrans, (point=pt, grad=nothing))
        temp = temp .+ sub(point, locgrad, args) .* wt
    end
    temp
end
pass_evalargs(::Type{<:ElementIntegral}) = true
raw_args(::Type{<:ElementIntegral}) = (1,)

struct Monomials{D,P,S} <: CompilationBlock end
@generated function (self::Monomials{D,P,S})(arg) where {D,P,S}
    exprs = [
        i <= P ? zero(eltype(arg)) : :(arg[$j] ^ $(i-P-1))
        for j in CartesianIndices(size(arg))
        for i in 1:(P+D+1)
    ]
    :(@inbounds SArray{$(Tuple{S...})}($(exprs...)))
end

struct PermuteDims{I} <: CompilationBlock end
@generated function (::PermuteDims{I})(arg) where I
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

struct Sum{D,S} <: CompilationBlock end
@generated function (self::Sum{D,S})(arg) where {D,S}
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

struct Contract{I,Ti,S} <: CompilationBlock end
@generated function (self::Contract{I,Ti,S})(args...) where {I,Ti,S}
    dims = Evaluables._sizedict(args, I)

    # dim_order maps an axis label to an arbitrary one-based index
    dim_order = Dict(axis => num for (num, axis) in enumerate(keys(dims)))

    # getind(indmap, i) unpacks the indices in i corresponding to the index labels in indmap
    getind = (indmap, i) -> (i[dim_order[ax]] for ax in indmap)

    sums = Vector{Expr}(undef, prod(S))
    for indices in product((1:n for n in values(dims))...)
        in_factors = map(enumerate(I)) do (i, ind)
            index = LinearIndices(size(args[i]))[getind(ind, indices)...]
            return :(args[$i][$index])
        end
        in_prod = :(*($(in_factors...)))
        out_index = LinearIndices(S)[getind(Ti, indices)...]
        sums[out_index] = isassigned(sums, out_index) ? :($(sums[out_index]) + $in_prod) : in_prod
    end

    :(@inbounds SArray{$(Tuple{S...})}($(sums...)))
end

end # module
