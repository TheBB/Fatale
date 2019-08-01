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
valueof(self::Zeros) = @SArray zeros(eltype(self), size(self)...)
