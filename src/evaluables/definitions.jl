# Return type of LocalCoords and GlobalCoords
const CoordsType{N, T} = NamedTuple{(:point, :grad), Tuple{StaticVector{N, T}, StaticMatrix{N, N, T}}}


"""
    LocalCoords(N, T=Float64)

Evaluable returning the local (parameter) N-dimensional coordinates of
the evaluation point, with element type T.
"""
struct LocalCoords{T} <: Evaluable{T}
    LocalCoords(N, T=Float64) = new{CoordsType{N, T}}()
end

@inline function (::LocalCoords{CoordsType{N,T}})(element, quadpt) where {N,T}
    igrad = SMatrix{N,N,T}(I)
    (point, grad) = loctrans(element)(quadpt, igrad)
    (point=point, grad=grad)
end


"""
    GlobalCoords(N, T=Float64)

Evaluable returning the global (physical) N-dimensional coordinates of
the evaluation point, with element type T.
"""
struct GlobalCoords{T} <: Evaluable{T}
    GlobalCoords(N, T=Float64) = new{CoordsType{N, T}}()
end

arguments(::GlobalCoords{CoordsType{N,T}}) where {N,T} = [LocalCoords(N,T)]

@inline function (::GlobalCoords{CoordsType{N,T}})(element, _, loc) where {N,T}
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
