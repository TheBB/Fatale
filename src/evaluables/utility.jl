# ==============================================================================
# Miscellaneous

function insertaxis(arg::Evaluable; left=0, right=0, at=nothing)
    left == 0 && right == 0 && at == nothing && return arg
    total = collect(Int, size(arg))
    at != nothing && insert!(total, at, 1)
    reshape(arg, ones(Int, left)..., total..., ones(Int, right)...)
end

flushleft(arg::Evaluable, totdims::Int) = insertaxis(arg; right=totdims-ndims(arg))
flushleft(arg::Evaluable, reference) = flushleft(arg, ndims(reference))
flushright(arg::Evaluable, totdims::Int) = insertaxis(arg; left=totdims-ndims(arg))
flushright(arg::Evaluable, reference) = flushright(arg, ndims(reference))


# ==============================================================================
# Broadcasting

# Dummy broadcastable object that wraps an Evaluable
struct Bcast{T<:ArrayEvaluable}
    wrapped :: T
end
Base.broadcastable(self::ArrayEvaluable) = Bcast(self)

# Broadcasting style that overrides everything else
struct BStyle <: Base.BroadcastStyle end
Base.BroadcastStyle(::Type{<:Bcast}) = BStyle()
Base.BroadcastStyle(::BStyle, ::Base.BroadcastStyle) = BStyle()

const materialize = Base.Broadcast.materialize
const Bcasted{T} = Base.Broadcast.Broadcasted{BStyle, Nothing, T}

_unwrap(bc::Base.Broadcast.Broadcasted) = _unwrap(materialize(bc))
_unwrap(bc::Bcast) = bc.wrapped
_unwrap(bc::Evaluable) = bc
_unwrap(bc::Union{AbstractArray,Real}) = Constant(bc)
_unwrap(bc::Ref) = _unwrap(bc[])
_unwrap(bc::Val{T}) where T = T

materialize(bc::Bcasted{typeof(+)}) = reduce(Add, map(_unwrap, bc.args))
materialize(bc::Bcasted{typeof(*)}) = reduce(Multiply, map(_unwrap, bc.args))

function materialize(bc::Bcasted{typeof(Base.literal_pow)})
    @assert bc.args[1] isa Ref{typeof(^)}
    Power(_unwrap(bc.args[2]), _unwrap(bc.args[3]))
end

function materialize(bc::Bcasted{typeof(sqrt)})
    @assert length(bc.args) == 1
    Sqrt(_unwrap(bc.args[1]))
end

function materialize(bc::Bcasted{typeof(/)})
    numerator = _unwrap(bc.args[1])
    denominator = reduce(Multiply, map(_unwrap, bc.args[2:end]))
    Multiply(numerator, Reciprocal(denominator))
end


# ==============================================================================
# Methods to other functions

function Base.:*(left::Evaluable, right::Evaluable)
    if ndims(left) == 0 || ndims(right) == 0
        return Multiply(left, right)
    end
    linds = Tuple(1:ndims(left))
    rinds = Tuple(ndims(left) - 1 .+ (1:ndims(right)))
    tinds = Tuple(flatten((linds[1:end-1], rinds[2:end])))
    Contract(left, right, linds, rinds, tinds)
end

Base.:-(self::Evaluable) = Negate(self)
Base.:-(left::Evaluable, right) = left + (-right)
Base.:+(left::Evaluable, right::Evaluable) = Add(left, right)
Base.:+(left::Evaluable, right) = Add(left, right)
Base.:+(left, right::Evaluable) = Add(left, right)

Base.getindex(self::Evaluable, index...) = GetIndex(self, index...)

Base.inv(self::Evaluable) = Inv(self)

function Base.reshape(self::Evaluable, newsize...)
    newsize = collect(Any, newsize)
    if (colon_index = findfirst(==(:), newsize)) != nothing
        in_length = prod(size(self))
        out_length = all(==(:), newsize) ? 1 : prod(filter(!(==(:)), newsize))
        @assert in_length % out_length == 0
        newsize[colon_index] = div(in_length, out_length)
    end
    @assert all(k != (:) for k in newsize)
    @assert prod(newsize) == prod(size(self))
    Reshape(self, newsize...)
end

function Base.adjoint(self::ArrayEvaluable)
    @assert eltype(self) <: Real
    transpose(self)
end
function Base.transpose(self::ArrayEvaluable)
    ndims(self) == 1 && return insertaxis(self; left=1)
    @assert ndims(self) == 2
    permutedims(self, (2, 1))
end
Base.permutedims(self::ArrayEvaluable, perm) = PermuteDims(self, perm)

Base.sum(self::Evaluable; dims=:, collapse=false) = Sum(self, dims, collapse)

LinearAlgebra.dot(left::ArrayEvaluable, right::ArrayEvaluable) = sum(left .* right; collapse=true)
LinearAlgebra.dot(left::ArrayEvaluable, right) = sum(left .* Constant(right); collapse=true)
LinearAlgebra.dot(left, right::ArrayEvaluable) = sum(Constant(left) .* right; collapse=true)

LinearAlgebra.normalize(vec) = vec ./ norm(vec, 2)

function LinearAlgebra.norm(self::ArrayEvaluable, p::Real=2)
    @assert p == 2
    sqrt.(dot(self, self))
end

LinearAlgebra.norm_sqr(self::ArrayEvaluable) = dot(self, self)


# ==============================================================================
# Convenience constructors

local_transform() = ElementData{_Transform}(:loctrans)
global_transform() = ElementData{_Transform}(:globtrans)
element_index(n) = ElementData{_Array}(:index; size=(n,), eltype=Int)

input_coords() = Argument{_Coords}(:point)

local_coords(n) = ApplyTrans(local_transform(), input_coords(), n)
local_point(n) = GetProperty(local_coords(n), :point)
local_grad(n) = GetProperty(local_coords(n), :grad)

global_coords(n) = ApplyTrans(global_transform(), local_coords(n), n)
global_point(n) = GetProperty(global_coords(n), :point)
global_grad(n) = GetProperty(global_coords(n), :grad)

function normal(geom)
    @assert ndims(geom) == 1
    lgrad = grad(geom, size(geom, 1))
    size(geom, 1) == 1 && return normalize(lgrad[:,end])
    G = lgrad[:, 1:end-1]
    n = lgrad[:, end]
    normalize(n - G * inv(G' * G) * G' * n)
end


# ==============================================================================
# Outer constructors

# Add: Ensure that constants accumulate on the left, and simplify them there
Add(self::Evaluable) = self
Add(left::Evaluable, right::Evaluable) = Add((left, right))
Add(left::Evaluable, right) = Add((Constant(right), left))
Add(left, right::Evaluable) = Add((Constant(left), right))
Add(left::Evaluable, right::Constant) = Add((right, left))
Add(left::Constant, right::Constant) = Constant(left.value .+ right.value)
Add(left::Add, right::Evaluable) = Add((left.args..., right))
Add(left::Evaluable, right::Add) = Add((right.args..., left))
Add(left::Constant, right::Add) = Add(right, left)

function Add(left::Add, right::Constant)
    if left.args[1] isa Constant
        return Add((left.args[1] .+ right, left.args[2:end]...))
    end
    Add((right, left.args...))
end

function Add(left::Add, right::Add)
    if left.args[1] isa Constant && right.args[1] isa Constant
        return Add((left.args[1] .+ right.args[1], left.args[2:end]..., right.args[2:end]...))
    end
    if right.args[1] isa Constant
        return Add((right.args[1], left.args..., right.args[2:end]...))
    end
    Add((left.args..., right.args...))
end

Constant(value::Real) = Constant(Scalar(value))
Constant(value::AbstractArray) = Constant(SArray{Tuple{size(value)...}, eltype(value)}(value))

Contract(left::ArrayEvaluable, right::ArrayEvaluable, lind::Dims, rind::Dims, target::Dims) =
    Contract((left, right), (lind, rind), target)

function Contract(left::Zeros, right::ArrayEvaluable, lind::Dims, rind::Dims, target::Dims)
    dims = _sizedict((left, right), (lind, rind))
    newsize = Tuple(dims[i] for i in target)
    newtype = promote_type(eltype(left), eltype(right))
    Zeros(newtype, newsize...)
end
Contract(left::ArrayEvaluable, right::Zeros, lind::Dims, rind::Dims, target::Dims) =
    Contract(right, left, rind, lind, target)

Inv(self::Constant) = Constant(inv(self.value))

# Multiply: Ensure that constants accumulate on the left, and simplify them there
Multiply(self::Evaluable) = self
Multiply(left::Evaluable, right::Evaluable) = Multiply((left, right))
Multiply(left::Evaluable, right) = Multiply((Constant(right), left))
Multiply(left, right::Evaluable) = Multiply((Constant(left), right))
Multiply(left::Evaluable, right::Constant) = Multiply((right, left))
Multiply(left::Constant, right::Constant) = Constant(left.value .* right.value)
Multiply(left::Multiply, right::Evaluable) = Multiply((left.args..., right))
Multiply(left::Evaluable, right::Multiply) = Multiply((right.args..., left))
Multiply(left::Constant, right::Multiply) = Multiply(right, left)

function Multiply(left::Multiply, right::Constant)
    if left.args[1] isa Constant
        return Multiply((left.args[1] .* right, left.args[2:end]...))
    end
    Multiply((right, left.args...))
end

function Multiply(left::Multiply, right::Multiply)
    if left.args[1] isa Constant && right.args[1] isa Constant
        return Multiply((left.args[1] .* right.args[1], left.args[2:end]..., right.args[2:end]...))
    end
    if right.args[1] isa Constant
        return Multiply((right.args[1], left.args..., right.args[2:end]...))
    end
    Multiply(left.args..., right.args...)
end

Multiply(left::Inflate, right::Inflate) =
    Inflate(Multiply(left.arg, right), left.indices, left.newsize, left.axis)
Multiply(left::Inflate, right::Evaluable) =
    Inflate(Multiply(left.arg, right), left.indices, left.newsize, left.axis)
Multiply(left::Evaluable, right::Inflate) =
    Inflate(Multiply(left, right.arg), right.indices, right.newsize, right.axis)

Negate(self::Constant) = Constant(-self.value)

PermuteDims(self::Constant, perm::Dims) = Constant(permutedims(self.value, perm))

Power(self::Constant, exp) = Constant(self.value .^ exp)
Power(self::Power, exp) = Power(self.arg, self.exp + exp)

Reciprocal(self::Constant) = Constant(1 ./ self.value)
Reciprocal(self::Reciprocal) = self.arg

Reshape(self::Reshape, args...) = Reshape(self.arg, args...)
Reshape(self::Constant, args...) = Constant(reshape(self.value, args...))
function Reshape(self::Inflate, newsize...)
    infaxis = self.axis
    newsize = collect(Int, newsize)

    if infaxis == 1
        new_infaxis = something(findfirst(!(==(1)), newsize))
    else
        before = prod(size(self)[1:infaxis-1])
        new_infaxis = something(findfirst(==(before), cumprod(newsize))) + 1
    end

    @assert newsize[new_infaxis] == size(self, infaxis)
    newsize[new_infaxis] = size(self.arg, infaxis)
    Inflate(reshape(self.arg, newsize...), self.indices, self.newsize, new_infaxis)
end

Sqrt(self::Constant) = Constant(sqrt.(self.val))
