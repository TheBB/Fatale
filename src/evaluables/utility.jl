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

Constant(value::Real) = Constant(Scalar(value))
Constant(value::AbstractArray) = Constant(SArray{Tuple{size(value)...}, eltype(value)}(value))

# Contract: Ensure that constants accumulate on the left, and simplify them there
function _do_contract(left, right, l, r, t)
    newsize = _contract_size((left, right), (l, r), t)
    newtype = promote_type(eltype(left), eltype(right))
    ret = @MArray zeros(newtype, newsize...)
    func = __Contract{(l,r),t}(ret)
    func(left, right)
end

function _fold_contract(left::Contract, right::Evaluable, l, r, t)
    # Axis renaming
    rename = Dict(zip(l, left.target))
    newinds = _newindices(left)
    nextind, state = iterate(newinds)

    function getind(ind)
        ind in keys(rename) && return rename[ind]
        rename[ind] = nextind
        nextind, state = iterate(newinds, state)
        rename[ind]
    end

    r = map(getind, r)
    t = map(getind, t)

    (left.args..., right), (left.indices..., r), t
end

function _fold_contract(left::Contract, right::Contract, l, r, t)
    r_to_m = Dict(zip(right.target, r))
    m_to_l = Dict(zip(l, left.target))

    newinds_m = _newindices(Contract((left, right), (l, r), t))
    newinds_l = _newindices(left)

    nextind_m, state_m = iterate(newinds_m)
    nextind_l, state_l = iterate(newinds_l)

    function getind_m(rind)
        rind in keys(r_to_m) && return r_to_m[rind]
        r_to_m[rind] = nextind_m
        nextind_m, state_m = iterate(newinds_m, state_m)
        r_to_m[rind]
    end

    function getind_l(mind)
        mind in keys(m_to_l) && return m_to_l[mind]
        m_to_l[mind] = nextind_l
        nextind_l, state_l = iterate(newinds_l, state_l)
        m_to_l[mind]
    end

    getind_r_to_m = getind_l ∘ getind_m

    new_rinds = Tuple(map(getind_r_to_m, ri) for ri in right.indices)
    new_target = map(getind_l, t)

    (left.args..., right.args...), (left.indices..., new_rinds...), new_target
end

function _collapse_outer(args, inds, target, i, j)
    if i > j
        i, j = j, i
    end
    other_inds = ∪(
        Set(flatten(inds[1:i-1])),
        Set(flatten(inds[i+1:j-1])),
        Set(flatten(inds[j+1:end])),
        Set(target)
    )
    newtarget = Tuple(i for i in (inds[i]..., inds[j]...) if i in other_inds)
    newarg = convert(Evaluable, _do_contract(valueof(args[i]), valueof(args[j]), inds[i], inds[j], newtarget))
    args = (newarg, args[1:i-1]..., args[i+1:j-1]..., args[j+1:end]...)
    inds = (newtarget, inds[1:i-1]..., inds[i+1:j-1]..., inds[j+1:end]...)
    args, inds
end

Contract(left::Evaluable, right::Evaluable, l, r, t) = Contract((left, right), (l, r), t)
Contract(left::Evaluable, right, l, r, t) = Contract((left, convert(Evaluable, right)), (l, r), t)
Contract(left, right::Evaluable, l, r, t) = Contract((convert(Evaluable, left), right), (l, r), t)
Contract(left::Evaluable, right::AbstractConstant, l, r, t) = Contract(right, left, r, l, t)
Contract(left::AbstractConstant, right::AbstractConstant, l, r, t) =
    convert(Evaluable, _do_contract(valueof(left), valueof(right), l, r, t))
Contract(left::Contract, right::Evaluable, l, r, t) = Contract(_fold_contract(left, right, l, r, t)...)
Contract(left::Evaluable, right::Contract, l, r, t) = Contract(_fold_contract(right, left, r, l, t)...)
Contract(left::AbstractConstant, right::Contract, l, r, t) = Contract(right, left, r, l, t)

function Contract(left::Contract, right::AbstractConstant, l, r, t)
    args, inds, target = _fold_contract(left, right, l, r, t)
    if args[1] isa AbstractConstant
        args, inds = _collapse_outer(args, inds, target, 1, length(args))
    end
    Contract(args, inds, target)
end

function Contract(left::Contract, right::Contract, l, r, t)
    args, inds, target = _fold_contract(left, right, l, r, t)
    k = length(left.args) + 1
    if left.args[1] isa AbstractConstant && right.args[1] isa AbstractConstant
        args, inds = _collapse_outer(args, inds, target, 1, k)
    elseif right.args[1] isa AbstractConstant
        args = (args[k], args[1:k-1]..., args[k+1:end]...)
        inds = (inds[k], inds[1:k-1]..., inds[k+1:end]...)
    end
    Contract(args, inds, target)
end

function Contract(left::Zeros, right::ArrayEvaluable, lind::Dims, rind::Dims, target::Dims)
    newsize = _contract_size((left, right), (lind, rind), target)
    newtype = promote_type(eltype(left), eltype(right))
    Zeros(newtype, newsize...)
end
Contract(left::ArrayEvaluable, right::Zeros, lind::Dims, rind::Dims, target::Dims) =
    Contract(right, left, rind, lind, target)

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

Sqrt(self::AbstractConstant) = convert(Evaluable, sqrt.(valueof(self)))
