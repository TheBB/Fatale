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
    _trans :: Evaluable{AbstractTransform}
    _point :: Evaluable{<:Coords}
    ApplyTrans(trans, point, N, T=Float64) = new{N, T}(trans, point)
end

arguments(self::ApplyTrans) = [self._trans, self._point]

@inline function (::ApplyTrans)(_, trans, point)
    (point, grad) = trans(point.point, point.grad)
    (point=point, grad=grad)
end


"""
    GetProperty{S, T}(arg)

Evaluable accessing a field of *arg* named S, with type T.
"""
struct GetProperty{S, T} <: Evaluable{T}
    arg :: Evaluable
end

arguments(self::GetProperty) = [self.arg]

@generated (::GetProperty{S})(_, arg) where S = quote
    @_inline_meta
    arg.$S
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

        any(arg isa Zeros for arg in args) && return Zeros(eltype(rtype), target_size...)

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
    GetIndex(arg, index...)

An evaluable returning a view into another array.
"""
struct GetIndex{I,T} <: Evaluable{T}
    arg :: Evaluable

    function GetIndex(arg, index...)
        @assert length(index) == ndims(arg)
        newsize = Tuple(s for (s,i) in zip(size(arg), index) if i isa Colon)
        rtype = marray(newsize, eltype(arg))
        new{Tuple{index...}, rtype}(arg)
    end
end

arguments(self::GetIndex) = [self.arg]

@generated (::GetIndex{I})(_, arg) where I = quote
    @_inline_meta
    uview(arg, $(I.parameters...))
end


"""
    Inv(arg)

An evaluable that computes the inverse of the two-dimensional argument
*arg*.
"""
struct Inv{T} <: Evaluable{T}
    arg :: Evaluable
    storage :: T

    function Inv(arg::Evaluable)
        @assert ndims(arg) == 2
        @assert size(arg, 1) == size(arg, 2)
        @assert size(arg, 1) < 4
        rtype = marray(size(arg), eltype(arg))
        new{rtype}(arg, rtype(undef))
    end
end

arguments(self::Inv) = [self.arg]

@generated function (self::Inv)(_, arg)
    dims = size(self, 1)
    T = eltype(self)
    if dims == 1
        quote
            self.storage[1,1] = $(one(T)) / arg[1,1]
            self.storage
        end
    elseif dims == 2
        quote
            self.storage[1,1] = arg[2,2]
            self.storage[2,2] = arg[1,1]
            self.storage[1,2] = -arg[1,2]
            self.storage[2,1] = -arg[2,1]
            self.storage ./= (arg[1,1] * arg[2,2] - arg[1,2] * arg[2,1])
            self.storage
        end
    elseif dims == 3
        quote
            self.storage[1,1] = arg[2,2] * arg[3,3] - arg[2,3] * arg[3,2]
            self.storage[2,1] = arg[2,3] * arg[3,1] - arg[2,1] * arg[3,3]
            self.storage[3,1] = arg[2,1] * arg[3,2] - arg[2,2] * arg[3,1]
            self.storage[1,2] = arg[1,3] * arg[3,2] - arg[1,2] * arg[3,3]
            self.storage[2,2] = arg[1,1] * arg[3,3] - arg[1,3] * arg[3,1]
            self.storage[3,2] = arg[1,2] * arg[3,1] - arg[1,1] * arg[3,2]
            self.storage[1,3] = arg[1,2] * arg[2,3] - arg[1,3] * arg[2,2]
            self.storage[2,3] = arg[1,3] * arg[2,1] - arg[1,1] * arg[2,3]
            self.storage[3,3] = arg[1,1] * arg[2,2] - arg[1,2] * arg[2,1]
            self.storage ./= (
                arg[1,1] * self.storage[1,1] +
                arg[1,2] * self.storage[2,1] +
                arg[1,3] * self.storage[3,1]
            )
            self.storage
        end
    end
end


"""
    Monomials(arg, degree, padding=0)

Computes all monomials of *arg* up to *degree*, with *padding* leading
zeros, yielding an array of size 

    (size(arg)..., padding + degree + 1).
"""
struct Monomials{D, P, T} <: Evaluable{T}
    arg :: Evaluable
    storage :: T

    function Monomials(arg::Evaluable, degree::Int, padding::Int)
        newsize = (size(arg)..., padding + degree + 1)
        rtype = marray(newsize, eltype(arg))
        new{degree, padding, rtype}(arg, rtype(undef))
    end
end

Monomials(arg, degree) = Monomials(arg, degree, 0)

arguments(self::Monomials) = [self.arg]

@generated function (self::Monomials{D, P})(_, arg) where {D, P}
    colons = [Colon() for _ in 1:ndims(self)-1]
    codes = [
        :(self.storage[$(colons...), $(P+i+1)] .= self.storage[$(colons...), $(P+i)] .* arg)
        for i in 1:D
    ]

    quote
        @_inline_meta
        self.storage[$(colons...), 1:$P] .= $(zero(eltype(self)))
        self.storage[$(colons...), $(P+1)] .= $(one(eltype(self)))
        $(codes...)
        self.storage
    end
end


"""
    Negate(arg)

Negate the argument.
"""
struct Negate{T} <: Evaluable{T}
    arg :: Evaluable
    Negate(arg) = new{restype(arg)}(arg)
end

arguments(self::Negate) = [self.arg]

@inline (::Negate)(_, arg) = -arg


"""
    Product(args...)

Elementwise product of arguments.
"""
struct Product{T} <: Evaluable{T}
    args :: Vector{Evaluable}
    storage :: T

    function Product(args...)
        rsize = broadcast_shape(map(size, args)...)
        rtype = marray(rsize, reduce(promote_type, map(eltype, args)))
        any(arg isa Zeros for arg in args) && return Zeros(eltype(rtype), rsize...)
        new{rtype}(collect(args), rtype(undef))
    end
end

arguments(self::Product) = self.args

# Generated to avoid allocating when splatting in .*(args...)
@generated function (self::Product)(_, args...)
    argcodes = [:(args[$i]) for i in 1:length(args)]
    quote
        self.storage .= $(zero(eltype(self)))
        self.storage .= .*($(argcodes...))
        self.storage
    end
end


"""
    Reshape(arg, size...)

Reshape *arg* to a new size.
"""
struct Reshape{T} <: Evaluable{T}
    arg :: Evaluable
    
    function Reshape(arg, newsize...)
        newsize = collect(Any, newsize)
        if (colon_index = findfirst(==(:), newsize)) != nothing
            in_length = prod(size(arg))
            out_length = all(==(:), newsize) ? 1 : prod(filter(!(==(:)), newsize))
            @assert in_length % out_length == 0
            newsize[colon_index] = div(in_length, out_length)
        end
        @assert all(k != (:) for k in newsize)
        rtype = sarray(newsize, eltype(arg))
        new{rtype}(arg)
    end
end

arguments(self::Reshape) = [self.arg]

@generated (self::Reshape)(_, arg) = :(reshape(arg, $(size(self)...)))


"""
    Sum(args...)

Elementwise sum of arguments.
"""
struct Sum{T} <: Evaluable{T}
    args :: Vector{Evaluable}
    storage :: T

    function Sum(args...)
        rsize = broadcast_shape(map(size, args)...)
        rtype = marray(rsize, reduce(promote_type, map(eltype, args)))
        new{rtype}(collect(args), rtype(undef))
    end
end

arguments(self::Sum) = self.args

# Generated to avoid allocating when splatting in .+(args...)
@generated function (self::Sum)(_, args...)
    argcodes = [:(args[$i]) for i in 1:length(args)]
    quote
        self.storage .= $(zero(eltype(self)))
        self.storage .= .+($(argcodes...))
        self.storage
    end
end


"""
    Zeros(T=Float64, size...)

Return a constant zero array of the given size and type.
"""
struct Zeros{T} <: Evaluable{T}
    Zeros(k::Type, size::Int...) = new{sarray(size, k)}()
end

Zeros(size::Int...) = Zeros(Float64, size...)

@generated (::Zeros{T})(_) where T = quote
    @SArray zeros($(eltype(T)), $(size(T)...))
end
