"""
    Constant(v)

An evaluable returning the constant array object *v*.
"""
struct Constant <: AbstractConstant
    value :: SArray
end

Base.eltype(self::Constant) = eltype(self.value)
Base.ndims(self::Constant) = ndims(self.value)
Base.size(self::Constant) = size(self.value)
valueof(self::Constant) = self.value

codegen(self::Constant) = __Constant(self.value)
struct __Constant{T}
    val :: T
end
@inline (self::__Constant)() = self.val


"""
    OneTo(stop)

Return a SOneTo object.
"""
struct OneTo <: AbstractConstant
    stop :: Int
end

Base.eltype(::OneTo) = Int
Base.size(self::OneTo) = (self.stop,)
valueof(self::OneTo) = SOneTo(self.stop)

codegen(self::OneTo) = __OneTo{self.stop}()
struct __OneTo{S} end
@inline (::__OneTo{S})() where S = SOneTo(S)


"""
    FUnitRange(start, stop)

An evaluable returning a SUnitRange object.
"""
struct FUnitRange <: AbstractConstant
    start :: Int
    stop :: Int
end

Base.eltype(self::FUnitRange) = Int
Base.size(self::FUnitRange) = (self.stop - self.start + 1,)
valueof(self::FUnitRange) = SUnitRange(self.start, self.stop)

codegen(self::FUnitRange) = __FUnitRange{self.start, self.stop}()
struct __FUnitRange{S,E} end
@inline (::__FUnitRange{S,E})() where {S,E}  = SUnitRange(S,E)


"""
    Zeros(T=Float64, dims...)

Return a constant zero array of the given size and type.
"""
struct Zeros <: AbstractConstant
    dims :: Dims
    eltype :: DataType
    Zeros(eltype::Type, dims::Int...) = new(dims, eltype)
end

Zeros(dims::Int...) = Zeros(Float64, dims...)
Base.eltype(self::Zeros) = self.eltype
Base.size(self::Zeros) = self.dims
valueof(self::Zeros) = zero(SArray{Tuple{size(self)...}, eltype(self), ndims(self), length(self)})

codegen(self::Zeros) = __Zeros{SArray{Tuple{size(self)...}, eltype(self), ndims(self), length(self)}}()
struct __Zeros{T} end
@inline (self::__Zeros{T})() where T = zero(T)
