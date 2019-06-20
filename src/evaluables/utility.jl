function Base.getproperty(self::Evaluable{T}, v::Symbol) where T<:NamedTuple
    index = findfirst(x->x==v, T.parameters[1])
    index == nothing && return getfield(self, v)
    rtype = T.parameters[2].parameters[index]
    GetProperty{v, rtype}(self)
end


# ==============================================================================
# Type constructors

staticarray(size, eltype, root) = root{Tuple{size...}, eltype, length(size), prod(size)}
marray(size, eltype) = staticarray(size, eltype, MArray)
sarray(size, eltype) = staticarray(size, eltype, SArray)


# ==============================================================================
# Methods to other functions

function Base.:*(left::Evaluable, right::Evaluable)
    linds = Tuple(1:ndims(left))
    rinds = Tuple(ndims(left) - 1 .+ (1:ndims(right)))
    tinds = Tuple(flatten((linds[1:end-1], rinds[2:end])))
    Contract((left, right), (linds, rinds), tinds)
end

Base.:-(self::Evaluable) = Negate(self)

Base.inv(self::Evaluable) = Inv(self)

Base.reshape(self::Evaluable, args...) = Reshape(self, args...)


# ==============================================================================
# Convenience constructors

local_transform() = ElementData{:loctrans, AbstractTransform}()
global_transform() = ElementData{:globtrans, AbstractTransform}()

input_coords() = Argument{:point,Coords}()

local_coords(n) = ApplyTrans(local_transform(), input_coords(), n)
local_point(n) = local_coords(n).point
local_grad(n) = local_coords(n).grad

global_coords(n) = ApplyTrans(global_transform(), local_coords(n), n)
global_point(n) = global_coords(n).point
global_grad(n) = global_coords(n).grad
