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

_checksize(_, ::Type{Int}) = nothing
_checksize(_, ::Type{Base.Slice{SOneTo{s}}}) where s = s

function Base.size(::Type{<:SubArray{T,N,<:StaticArray{S},I}}) where {T,N,S,I}
    Tuple(filter(!isnothing, map(_checksize, S.parameters, I.parameters)))
end


# ==============================================================================
# Methods to other functions

function Base.:*(left::Evaluable, right::Evaluable)
    linds = Tuple(1:ndims(left))
    rinds = Tuple(ndims(left) - 1 .+ (1:ndims(right)))
    tinds = Tuple(flatten((linds[1:end-1], rinds[2:end])))
    Contract(left, right, linds, rinds, tinds)
end

Base.:-(self::Evaluable) = Negate(self)

Base.broadcasted(::typeof(+), args::Evaluable...) = Sum(args...)
Base.broadcasted(::typeof(*), args::Evaluable...) = Product(args...)

Base.getindex(self::Evaluable, index...) = GetIndex(self, index...)

Base.inv(self::Evaluable) = Inv(self)

Base.reshape(self::Evaluable, args...) = Reshape(self, args...)
Base.reshape(self::Reshape, args...) = reshape(self.arg, args...)
Base.reshape(self::Constant, args...) = Constant(reshape(self.value, args...))


# ==============================================================================
# Convenience constructors

local_transform() = ElementData{_Transform}(:loctrans)
global_transform() = ElementData{_Transform}(:globtrans)

input_coords() = Argument{_Coords}(:point)

local_coords(n) = ApplyTrans(local_transform(), input_coords(), n)
local_point(n) = GetProperty(local_coords(n), :point)
local_grad(n) = GetProperty(local_coords(n), :grad)

global_coords(n) = ApplyTrans(global_transform(), local_coords(n), n)
global_point(n) = GetProperty(global_coords(n), :point)
global_grad(n) = GetProperty(global_coords(n), :grad)


# ==============================================================================
# Outer constructors

Contract(left::Evaluable{_Array}, right::Evaluable{_Array}, lind::Dims, rind::Dims, target::Dims) =
    Contract((left, right), (lind, rind), target)

function Contract(left::Zeros, right::Evaluable{_Array}, lind::Dims, rind::Dims, target::Dims)
    dims = _sizedict((left, right), (lind, rind))
    newsize = Tuple(dims[i] for i in target)
    newtype = promote_type(eltype(left), eltype(right))
    Zeros(newtype, newsize...)
end
Contract(left::Evaluable{_Array}, right::Zeros, lind::Dims, rind::Dims, target::Dims) =
    Contract(right, left, rind, lind, target)
