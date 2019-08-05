# The Add and Multiply evaluables use almost the exact same interface and logic,
# here combined into a "commutative arithmetic" type
abstract type CommArith <: ArrayEvaluable end

# This functions as a shared inner constructor
function CommArith(cons, args::Vector{<:ArrayEvaluable})
    length(args) == 1 && return args[1]

    # As a means of simplification, all constants should accumulate
    # and condense into the left argument
    @assert all(!(arg isa AbstractConstant) for arg in args[2:end])

    dims = broadcast_shape(map(size, args)...)
    cons(collect(Evaluable, args), dims)
end

arguments(self::CommArith) = self.args
size(self::CommArith) = self.dims


# Outer constructors with purpose:
# - CommAriths of other CommAriths become bigger CommAriths
# - Constants accumulate in the leftmost argument, if any

# One argument is a no-op
CommArith(cons, arg::ArrayEvaluable) = arg

# Basic conversion of arguments
CommArith(cons::Type{<:CommArith}, left::ArrayEvaluable, right::ArrayEvaluable) = CommArith(cons, [left, right])
CommArith(cons::Type{<:CommArith}, left::ArrayEvaluable, right) = CommArith(cons, left, convert(Evaluable, right))
CommArith(cons::Type{<:CommArith}, left, right::ArrayEvaluable) = CommArith(cons, convert(Evaluable, left), right)

# Move constants on the left
CommArith(cons::Type{<:CommArith}, left::ArrayEvaluable, right::AbstractConstant) = CommArith(cons, right, left)

# Immediately compute constants
CommArith(cons::Type{<:CommArith}, left::AbstractConstant, right::AbstractConstant) =
    convert(Evaluable, _compute(cons, valueof(left), valueof(right)))

# Combine CommAriths into bigger CommAriths
CommArith(cons::Type{T}, left::T, right::ArrayEvaluable) where T<:CommArith = CommArith(cons, [left.args..., right])
CommArith(cons::Type{T}, left::ArrayEvaluable, right::T) where T<:CommArith = CommArith(cons, [right.args..., left])

# Compute constants that occur due to CommArith combination (one)
CommArith(cons::Type{T}, left::AbstractConstant, right::T) where T<:CommArith = CommArith(cons, right, left)
function CommArith(cons::Type{T}, left::T, right::AbstractConstant) where T<:CommArith
    if left.args[1] isa AbstractConstant
        ncons = convert(Evaluable, _compute(cons, valueof(left.args[1]), valueof(right)))
        return CommArith(cons, [ncons, left.args[2:end]...])
    end
    CommArith(cons, [right, left.args...])
end

# Compute constants that occur due to CommArith combination (two)
function CommArith(cons::Type{T}, left::T, right::T) where T<:CommArith
    if left.args[1] isa AbstractConstant && right.args[1] isa AbstractConstant
        ncons = convert(Evaluable, _compute(cons, valueof(left.args[1]), valueof(right.args[1])))
        return CommArith(cons, [ncons, left.args[2:end]..., right.args[2:end]...])
    end
    if right.args[1] isa AbstractConstant
        return CommArith(cons, [right.args[1], left.args..., right.args[2:end]...])
    end
    CommArith(cons, [left.args..., right.args...])
end

# Move Zeros to the left argument
# The actual handling of zero is different for each subtype, see below
CommArith(cons::Type{<:CommArith}, left::Zeros, right::ArrayEvaluable) = cons(left, right)
CommArith(cons::Type{<:CommArith}, left::ArrayEvaluable, right::Zeros) = CommArith(cons, right, left)

# Move Inflate to the left argument
# The actual handling of inflate is different for each subtype, see below
CommArith(cons::Type{<:CommArith}, left::Inflate, right::ArrayEvaluable) = cons(left, right)
CommArith(cons::Type{<:CommArith}, left::ArrayEvaluable, right::Inflate) = CommArith(cons, right, left)

# Ambiguity resolution

# Inflate x Inflate -> resolve the left one first (completely arbitrary)
CommArith(cons::Type{<:CommArith}, left::Inflate, right::Inflate) =
    invoke(CommArith, Tuple{Type{cons},Inflate,ArrayEvaluable}, cons, left, right)

# Inflate x Constant -> resolve the inflation first
CommArith(cons::Type{<:CommArith}, left::Inflate, right::AbstractConstant) =
    invoke(CommArith, Tuple{Type{cons},Inflate,ArrayEvaluable}, cons, left, right)
CommArith(cons::Type{<:CommArith}, left::AbstractConstant, right::Inflate) =
    invoke(CommArith, Tuple{Type{cons},Inflate,ArrayEvaluable}, cons, right, left)


"""
    Add(left, right)

Elementwise sum of arguments.
"""
struct Add <: CommArith
    args :: Vector{Evaluable}
    dims :: Dims
    Add(args::Vector{Evaluable}, dims::Dims) = new(args, dims)
end

Add(args...) = CommArith(Add, args...)
_compute(::Type{Add}, left, right) = left .+ right
codegen(self::Add) = CplCommArith{:.+}()

function Add(left::Zeros, right::ArrayEvaluable)
    # TODO: Add a promote_type evaluable
    @assert eltype(left) == eltype(right)

    # TODO Add a repmat evaluable or something similar
    newsize = broadcast_shape(size(left), size(right))
    @assert newsize == size(right)

    right
end


"""
    Multiply(left, right)

Elementwise product of arguments.
"""
struct Multiply <: CommArith
    args :: Vector{Evaluable}
    dims :: Dims
    Multiply(args::Vector{Evaluable}, dims::Dims) = new(args, dims)
end

Multiply(args...) = CommArith(Multiply, args...)
_compute(::Type{Multiply}, left, right) = left .* right
codegen(self::Multiply) = CplCommArith{:.*}()

function Multiply(left::Zeros, right::ArrayEvaluable)
    newsize = broadcast_shape(size(left), size(right))
    newtype = promote_type(eltype(left), eltype(right))
    Zeros(newtype, newsize...)
end

# TODO: This is not sufficient
Multiply(left::Inflate, right::ArrayEvaluable) =
    Inflate(Multiply(left.arg, right), left.indices, left.newsize, left.axis)
