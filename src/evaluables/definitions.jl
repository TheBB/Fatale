const Coords{N, T} = NamedTuple{(:point, :grad), Tuple{StaticVector{N, T}, StaticMatrix{N, N, T}}}


struct Argument{V,T} <: Evaluable{T} end

@generated (::Argument{V})(input) where V = quote
    @_inline_meta
    input.$V
end


"""
    ElementData{sym, T}(args...)

An evaluable that accesses element data named `sym` of type T. Some
standard names are defined:

- :loctrans -> the local parameter transformation
- :globtrans -> the global physical transformation

You can use others so long as you know that the element type supports
them, that is, there is a method of Fatale.Elements.ElementData

    elementdata(::ElementType, ::Val{sym}, args...) :: T
"""
struct ElementData{V, T, A<:Tuple} <: Evaluable{T}
    args :: A
    ElementData{V,T}(a...) where {V,T} = new{V,T,typeof(a)}(a)
end

@inline (self::ElementData{V})(input) where V = elementdata(input.element, Val(V), self.args...)


"""
    ApplyTrans(trans, point, N, T=Float64)

Apply `trans` to `point`, producing an N-dimensional coordinate of
element type T.
"""
struct ApplyTrans{N, T} <: Evaluable{Coords{N, T}}
    _trans :: Evaluable{Transform}
    _point :: Evaluable{<:Coords}
    ApplyTrans(trans, point, N, T=Float64) = new{N, T}(trans, point)
end

arguments(self::ApplyTrans) = [self._trans, self._point]

@inline function (::ApplyTrans)(_, trans, point)
    (point, grad) = trans(point.point, point.grad)
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

@generated function (::GetProperty{S})(_, arg) where S
    quote
        @_inline_meta
        arg.$S
    end
end


"""
    Contract((args...), (inds...), target_inds)

Compute a fully unrolled tensor contraction.
"""
struct Contract{Inds, Tinds, T} <: Evaluable{T}
    args :: Vector{Evaluable}
    storage :: T

    function Contract(args, inds, tinds)
        @assert length(args) == length(inds)
        @assert all(ndims(arg) == length(ind) for (arg, ind) in zip(args, inds))

        dims = _sizedict(args, inds)
        for (arg, ind) in zip(args, inds)
            @assert all(size(arg, i) == dims[ind[i]] for i in 1:ndims(arg))
        end

        target_size = Tuple(dims[i] for i in tinds)
        rtype = marray(target_size, reduce(promote_type, map(eltype, args)))

        Inds = Tuple{(Tuple{ind...} for ind in inds)...}
        Tinds = Tuple{tinds...}

        new{Inds, Tinds, rtype}(collect(args), rtype(undef))
    end
end

arguments(self::Contract) = self.args

@generated function (self::Contract{Inds, Tinds})(_, args...) where {Inds, Tinds}
    inds = collect(collect(tp.parameters) for tp in Inds.parameters)
    tinds = collect(Tinds.parameters)
    dims = _sizedict(args, inds)
    dim_order = Dict(axis => num for (num, axis) in enumerate(keys(dims)))

    codes = Expr[]
    for indices in product((1:n for n in values(dims))...)
        inputs = [
            :(args[$i][$((indices[dim_order[ax]] for ax in ind)...)])
            for (i, ind) in enumerate(inds)
        ]
        product = :(*($(inputs...)))
        target = :(self.storage[$((indices[dim_order[ax]] for ax in tinds)...)])
        push!(codes, :($target += $product))
    end

    quote
        @_inline_meta
        self.storage .= $(zero(eltype(self)))
        $(codes...)
        self.storage
    end
end

_sizedict(args, inds) = OrderedDict(flatten(
    (k => v for (k, v) in zip(ind, size(arg)))
    for (arg, ind) in zip(args, inds)
))



"""
    Constant(v)

An evaluable returning the constant object *v*.
"""
struct Constant{T} <: Evaluable{T}
    value :: T
end

@inline (self::Constant)(_) = self.value

Base.eltype(self::Constant) = eltype(self.value)
Base.ndims(self::Constant) = ndims(self.value)
Base.size(self::Constant) = size(self.value)

# Constants with different underlying objects must be considered
# distinct. This overrides the behaviour of Evaluable.
Base.hash(self::Constant, x::UInt64) = hash(self.value, x)
Base.:(==)(l::Constant, r::Constant) = l.value == r.value


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

@generated function (self::Monomials{D})(_, arg) where {D}
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
