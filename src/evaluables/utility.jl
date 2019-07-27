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

asevaluable(self::Evaluable, _) = self
asevaluable(self::AbstractArray, _) = Constant(self)
asevaluable(self::Real, _) = Constant(self)
asevaluable(self::UnitRange, _) = self.start == 1 ? OneTo(self.stop) : FUnitRange(self.start, self.stop)
asevaluable(self::OneTo, _) = OneTo(self.stop)
asevaluable(self::Colon, len) = OneTo(len)

function asevaluable(self::OptimizedBlockEvaluable, _)
    ret = self.data
    for (axis, index, newsize) in enumerate(zip(self.indices, size(self)))
        ret = Inflate(ret, index, newsize, axis)
    end
    ret
end

Base.convert(::Type{Evaluable}, x) = asevaluable(x, nothing)
Base.convert(::Type{Evaluable}, x::Evaluable) = x
Base.convert(::Type{Evaluable}, ::Colon) = error("length must be specified to transform Colon to Evaluable")


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
_unwrap(bc::Ref) = _unwrap(bc[])
_unwrap(bc::Val{T}) where T = T
_unwrap(bc) = convert(Evaluable, bc)

materialize(bc::Bcasted{typeof(+)}) = reduce(Add, map(_unwrap, bc.args))
materialize(bc::Bcasted{typeof(*)}) = reduce(Multiply, map(_unwrap, bc.args))

function materialize(bc::Bcasted{typeof(^)})
    @assert length(bc.args) == 2
    Power(_unwrap(bc.args[1]), bc.args[2])
end

function materialize(bc::Bcasted{typeof(Base.literal_pow)})
    @assert bc.args[1] isa Ref{typeof(^)}
    @assert length(bc.args) == 3
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
    linds = collect(Int, 1:ndims(left))
    rinds = collect(Int, ndims(left) - 1 .+ (1:ndims(right)))
    tinds = vcat(linds[1:end-1], rinds[2:end])
    Contract(left, right, linds, rinds, tinds)
end

Base.:*(left::Evaluable, right) = left * convert(Evaluable, right)
Base.:*(left, right::Evaluable) = convert(Evaluable, left) * right

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
LinearAlgebra.dot(left::ArrayEvaluable, right) = sum(left .* convert(Evaluable, right); collapse=true)
LinearAlgebra.dot(left, right::ArrayEvaluable) = sum(convert(Evaluable, left) .* right; collapse=true)

LinearAlgebra.normalize(vec) = vec ./ norm(vec, 2)

function LinearAlgebra.norm(self::ArrayEvaluable, p::Real=2)
    @assert p == 2
    sqrt.(dot(self, self))
end

LinearAlgebra.norm_sqr(self::ArrayEvaluable) = dot(self, self)


# ==============================================================================
# Convenience constructors

argument(T, name, args...) = Funcall(T, :getfield, EvalArgs(), name, args...)
element() = argument(_Element, :element)
element_data(T, name, args...) = Funcall(T, :elementdata, element(), Val(name), args...)

local_transform() = element_data(_Transform, :loctrans)
global_transform() = element_data(_Transform, :globtrans)
element_index(n) = element_data(_Array, :index, Int, (n,))

_point(self::CoordsEvaluable) = Funcall(_Array, :getfield, self, :point, eltype(self), (ndims(self),))
_grad(self::CoordsEvaluable) = Funcall(_Array, :getfield, self, :grad, eltype(self), (ndims(self),ndims(self)))

local_coords(n) = argument(_Coords, :point, Float64, (n,))
local_point(n) = _point(local_coords(n))
local_grad(n) = _grad(local_coords(n))

global_coords(n) = ApplyTrans(global_transform(), local_coords(n))
global_point(n) = _point(global_coords(n))
global_grad(n) = _grad(global_coords(n))

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
Add(left::Evaluable, right) = Add((convert(Evaluable, right), left))
Add(left, right::Evaluable) = Add((convert(Evaluable, left), right))
Add(left::Evaluable, right::AbstractConstant) = Add((right, left))
Add(left::AbstractConstant, right::AbstractConstant) = convert(Evaluable, valueof(left) .+ valueof(right))
Add(left::Add, right::Evaluable) = Add((left.args..., right))
Add(left::Evaluable, right::Add) = Add((right.args..., left))
Add(left::AbstractConstant, right::Add) = Add(right, left)

function Add(left::Add, right::AbstractConstant)
    if left.args[1] isa AbstractConstant
        return Add(left.args[1] .+ right, left.args[2:end]...)
    end
    Add((right, left.args...))
end

function Add(left::Add, right::Add)
    if left.args[1] isa AbstractConstant && right.args[1] isa AbstractConstant
        return Add((left.args[1] .+ right.args[1], left.args[2:end]..., right.args[2:end]...))
    end
    if right.args[1] isa AbstractConstant
        return Add((right.args[1], left.args..., right.args[2:end]...))
    end
    Add((left.args..., right.args...))
end

Add(left::Evaluable, right::Zeros) = Add(right, left)
function Add(left::Zeros, right::Evaluable)
    # TODO: Add a promote_type evaluable
    @assert eltype(left) == eltype(right)

    # TODO Add a repmat evaluable or something similar
    newsize = broadcast_shape(size(left), size(right))
    @assert newsize == size(right)

    right
end

Constant(value::Real) = Constant(Scalar(value))
Constant(value::AbstractArray) = Constant(SArray{Tuple{size(value)...}, eltype(value)}(value))

Inv(self::Constant) = Constant(inv(self.value))

function Monomials(self::Constant, d, p)
    func = __Monomials(d, p, eltype(self), size(self))
    Constant(func(nothing, self.value))
end

# Multiply: Ensure that constants accumulate on the left, and simplify them there
Multiply(self::Evaluable) = self
Multiply(left::Evaluable, right::Evaluable) = Multiply((left, right))
Multiply(left::Evaluable, right) = Multiply((convert(Evaluable, right), left))
Multiply(left, right::Evaluable) = Multiply((convert(Evaluable, left), right))
Multiply(left::Evaluable, right::AbstractConstant) = Multiply(right, left)
Multiply(left::AbstractConstant, right::AbstractConstant) = convert(Evaluable, valueof(left) .* valueof(right))
Multiply(left::Multiply, right::Evaluable) = Multiply((left.args..., right))
Multiply(left::Evaluable, right::Multiply) = Multiply((right.args..., left))
Multiply(left::AbstractConstant, right::Multiply) = Multiply(right, left)

function Multiply(left::Multiply, right::AbstractConstant)
    if left.args[1] isa AbstractConstant
        return Multiply((left.args[1] .* right, left.args[2:end]...))
    end
    Multiply((right, left.args...))
end

function Multiply(left::Multiply, right::Multiply)
    if left.args[1] isa AbstractConstant && right.args[1] isa AbstractConstant
        return Multiply((left.args[1] .* right.args[1], left.args[2:end]..., right.args[2:end]...))
    elseif right.args[1] isa AbstractConstant
        return Multiply((right.args[1], left.args..., right.args[2:end]...))
    end
    Multiply(left.args..., right.args...)
end

Multiply(left::Evaluable, right::Zeros) = Multiply(right, left)
function Multiply(left::Zeros, right::Evaluable)
    newsize = broadcast_shape(size(left), size(right))
    newtype = promote_type(eltype(left), eltype(right))
    Zeros(newtype, newsize...)
end

# TODO: These are not sufficient
Multiply(left::Inflate, right::Inflate) =
    Inflate(Multiply(left.arg, right), left.indices, left.newsize, left.axis)
Multiply(left::Inflate, right::Evaluable) =
    Inflate(Multiply(left.arg, right), left.indices, left.newsize, left.axis)
Multiply(left::Evaluable, right::Inflate) =
    Inflate(Multiply(left, right.arg), right.indices, right.newsize, right.axis)

Negate(self::AbstractConstant) = convert(Evaluable, -valueof(self))

PermuteDims(self::AbstractConstant, perm::Dims) = convert(Evaluable, permutedims(valueof(self), perm))

Power(self::AbstractConstant, exp) = convert(Evaluable, valueof(self) .^ exp)
Power(self::Power, exp) = Power(self.arg, self.exp + exp)

Reciprocal(self::AbstractConstant) = convert(Evaluable, 1 ./ valueof(self))
Reciprocal(self::Reciprocal) = self.arg

Reshape(self::Reshape, args...) = Reshape(self.arg, args...)
Reshape(self::AbstractConstant, args...) = convert(Evaluable, reshape(valueof(self), args...))

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

function Sum(self::Inflate, dims, collapse::Bool)
    # TODO: Temporary restriction
    @assert !(self.axis in dims)
    Inflate(sum(self.arg; dims=dims, collapse=collapse), self.indices, self.newsize, self.axis)
end

Sqrt(self::AbstractConstant) = convert(Evaluable, sqrt.(valueof(self)))
