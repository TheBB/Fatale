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

convert(::Type{Evaluable}, x) = asevaluable(x, nothing)
convert(::Type{Evaluable}, x::Evaluable) = x
convert(::Type{Evaluable}, ::Colon) = error("length must be specified to transform Colon to Evaluable")

staticarray(cons, size, eltype) = cons{Tuple{size...}, eltype, length(size), prod(size)}
staticarray(cons, x) = staticarray(cons, size(x), eltype(x))
sarray(args...) = staticarray(SArray, args...)
marray(args...) = staticarray(MArray, args...)


# ==============================================================================
# Broadcasting

# Dummy broadcastable object that wraps an Evaluable
struct Bcast{T<:ArrayEvaluable}
    wrapped :: T
end
broadcastable(self::ArrayEvaluable) = Bcast(self)

# Broadcasting style that overrides everything else
struct BStyle <: BroadcastStyle end
BroadcastStyle(::Type{<:Bcast}) = BStyle()
BroadcastStyle(::BStyle, ::BroadcastStyle) = BStyle()

const Bcasted{T} = Broadcasted{BStyle, Nothing, T}

_unwrap(bc::Broadcasted) = _unwrap(materialize(bc))
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

function materialize(bc::Bcasted{typeof(literal_pow)})
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

function *(left::Evaluable, right::Evaluable)
    if ndims(left) == 0 || ndims(right) == 0
        return Multiply(left, right)
    end
    linds = collect(Int, 1:ndims(left))
    rinds = collect(Int, ndims(left) - 1 .+ (1:ndims(right)))
    tinds = vcat(linds[1:end-1], rinds[2:end])
    Contract(left, right, linds, rinds, tinds)
end

*(left::Evaluable, right) = left * convert(Evaluable, right)
*(left, right::Evaluable) = convert(Evaluable, left) * right

-(self::Evaluable) = Negate(self)
-(left::Evaluable, right) = left + (-right)
+(left::Evaluable, right::Evaluable) = Add(left, right)
+(left::Evaluable, right) = Add(left, right)
+(left, right::Evaluable) = Add(left, right)

getindex(self::Evaluable, index...) = GetIndex(self, index...)

inv(self::Evaluable) = Inv(self)

function reshape(self::Evaluable, newsize...)
    newsize = collect(Any, newsize)
    if (colon_index = findfirst(==(:), newsize)) != nothing
        in_length = prod(size(self))
        out_length = all(==(:), newsize) ? 1 : prod(filter(!(==(:)), newsize))
        @assert in_length % out_length == 0
        newsize[colon_index] = div(in_length, out_length)
    end
    @assert all(k != (:) for k in newsize)
    @assert prod(newsize) == prod(size(self))
    Tuple(newsize) == size(self) && return self
    Reshape(self, newsize...)
end

function adjoint(self::ArrayEvaluable)
    @assert eltype(self) <: Real
    transpose(self)
end
function transpose(self::ArrayEvaluable)
    ndims(self) == 1 && return insertaxis(self; left=1)
    @assert ndims(self) == 2
    permutedims(self, (2, 1))
end
permutedims(self::ArrayEvaluable, perm) = PermuteDims(self, perm)

sum(self::Evaluable; dims=:, collapse=false) = Sum(self, dims, collapse)

dot(left::ArrayEvaluable, right::ArrayEvaluable) = sum(left .* right; collapse=true)
dot(left::ArrayEvaluable, right) = sum(left .* convert(Evaluable, right); collapse=true)
dot(left, right::ArrayEvaluable) = sum(convert(Evaluable, left) .* right; collapse=true)

normalize(vec) = vec ./ norm(vec, 2)

function norm(self::ArrayEvaluable, p::Real=2)
    @assert p == 2
    sqrt.(dot(self, self))
end

norm_sqr(self::ArrayEvaluable) = dot(self, self)


# ==============================================================================
# Convenience constructors

local_transform() = ElementData{_Transform}(:loctrans)
global_transform() = ElementData{_Transform}(:globtrans)
element_index(n) = ElementData{_Array}(:index; eltype=Int, size=(n,))

local_coords(n) = EvalArg{_Coords}(:coords; eltype=Float64, ndims=n)
local_point(n) = ExtractCoords(local_coords(n))
local_grad(n) = ExtractCoords(local_coords(n), 1)

global_coords(n) = ApplyTrans(global_transform(), local_coords(n))
global_point(n) = ExtractCoords(global_coords(n))
global_grad(n) = ExtractCoords(global_coords(n), 1)

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

Constant(value::Real) = Constant(Scalar(value))
Constant(value::AbstractArray) = Constant(sarray(value)(value))

Inv(self::Constant) = Constant(inv(self.value))

function Monomials(self::Constant, d, p)
    func = __Monomials(d, p, eltype(self), size(self))
    Constant(func(nothing, self.value))
end

Negate(self::AbstractConstant) = convert(Evaluable, -valueof(self))

PermuteDims(self::AbstractConstant, perm::Dims) = convert(Evaluable, permutedims(valueof(self), perm))

Power(self::AbstractConstant, exp) = convert(Evaluable, valueof(self) .^ exp)
Power(self::Power, exp) = Power(self.arg, self.exp + exp)

Reciprocal(self::AbstractConstant) = convert(Evaluable, 1 ./ valueof(self))
Reciprocal(self::Reciprocal) = self.arg

Reshape(self::Reshape, args...) = reshape(self.arg, args...)
Reshape(self::AbstractConstant, args...) = convert(Evaluable, reshape(valueof(self), args...))

function Reshape(self::Inflate, newsize...)
    infaxis = self.axis
    newsize = collect(Int, newsize)

    preproduct = prod(size(self)[1:infaxis])
    haystack = collect(zip(newsize, cumprod(newsize)))
    new_infaxis = something(findfirst(haystack) do (sz, pr)
        sz == size(self, infaxis) && pr == preproduct
    end)

    newsize[new_infaxis] = size(self.arg, infaxis)
    Inflate(reshape(self.arg, newsize...), self.indices, self.newsize, new_infaxis)
end

function Sum(self::Inflate, dims, collapse::Bool)
    # TODO: Temporary restriction
    @assert !(self.axis in dims)
    new_infaxis = self.axis
    collapse && (new_infaxis -= sum(dims .< self.axis))
    Inflate(sum(self.arg; dims=dims, collapse=collapse), self.indices, self.newsize, new_infaxis)
end

Sqrt(self::AbstractConstant) = convert(Evaluable, sqrt.(valueof(self)))
