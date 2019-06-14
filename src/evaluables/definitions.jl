# Return type of LocalCoords and GlobalCoords
const CoordsType{N, T} = NamedTuple{(:point, :grad), Tuple{StaticVector{N, T}, StaticMatrix{N, N, T}}}


"""
    LocalCoords(N, T=Float64)

Evaluable returning the local (parameter) N-dimensional coordinates of
the evaluation point, with element type T.
"""
struct LocalCoords{N, T} <: Evaluable{CoordsType{N, T}}
    LocalCoords(N, T=Float64) = new{N,T}()
end

@inline function (::LocalCoords{N,T})(element, quadpt) where {N,T}
    igrad = SMatrix{N,N,T}(I)
    (point, grad) = loctrans(element)(quadpt, igrad)
    (point=point, grad=grad)
end


"""
    GlobalCoords(N, T=Float64)

Evaluable returning the global (physical) N-dimensional coordinates of
the evaluation point, with element type T.
"""
struct GlobalCoords{N, T} <: Evaluable{CoordsType{N, T}}
    GlobalCoords(N, T=Float64) = new{N, T}()
end

arguments(::GlobalCoords{N,T}) where {N,T} = [LocalCoords(N,T)]

@inline function (::GlobalCoords{N,T})(element, _, loc) where {N,T}
    (point, grad) = globtrans(element)(loc.point, loc.grad)
    (point=point, grad=grad)
end


"""
    GetProperty{T, S}(arg)

Evaluable accessing a field of *arg* named S, with type T.
"""
struct GetProperty{S, T} <: Evaluable{T}
    arg :: Evaluable
end

arguments(self::GetProperty) = [self.arg]

@generated function (::GetProperty{S})(_, _, arg) where S
    quote
        @_inline_meta
        arg.$S
    end
end


"""
    Monomials(arg, degree)

Computes all monomials of *arg* up to *degree*, yielding an array of
size (size(arg)..., degree+1).
"""
struct Monomials{D, T} <: Evaluable{T}
    arg :: Evaluable
    storage :: T

    function Monomials(arg::Evaluable, degree::Int)
        newsize = (size(arg)..., degree + 1)
        rtype = marray(newsize, eltype(arg))
        new{degree, rtype}(arg, rtype(undef))
    end
end

arguments(self::Monomials) = [self.arg]

@generated function (self::Monomials{D})(_, _, arg) where {D}
    colons = [Colon() for _ in 1:ndims(self)-1]
    codes = [
        :(self.storage[$(colons...), $(i+1)] .= self.storage[$(colons...), $i] .* arg)
        for i in 1:D
    ]

    quote
        @_inline_meta
        self.storage[$(colons...), 1] .= $(one(eltype(self)))
        $(codes...)
        self.storage
    end
end
