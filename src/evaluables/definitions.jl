"""
    Argument{T}(name::Symbol)

Evaluable that obtains the evaluation argument named *name* of type *T*.
"""
struct Argument{T} <: Evaluable{T}
    name :: Symbol
end

codegen(self::Argument) = __Argument{self.name}()
struct __Argument{V} end
@generated (::__Argument{V})(input) where V = quote
    @_inline_meta
    input.$V
end


"""
    ElementData{V::Symbol, T}(args...)

An evaluable that accesses element data named *V* of type *T*. Some
standard names are defined:

- :loctrans -> the local parameter transformation
- :globtrans -> the global physical transformation

You can use others so long as you know that the element type supports
them, that is, there is a method of Fatale.Elements.elementdata

    elementdata(::ElementType, ::Val{sym}, args...) :: T
"""
struct ElementData{T} <: Evaluable{T}
    name :: Symbol
    args :: Tuple
    ElementData{T}(name, args...) where T = new{T}(name, args)
end

codegen(self::ElementData) = __ElementData{self.name}(self.args)
struct __ElementData{V,T}
    args :: T
    __ElementData{V}(args) where V = new{V,typeof(args)}(args)
end
@inline (self::__ElementData{V})(input) where V = elementdata(input.element, Val(V), self.args...)


"""
    ApplyTrans(trans, point, N, T=Float64)

Apply `trans` to `point`, producing an N-dimensional coordinate of
element type T.
"""
struct ApplyTrans <: Evaluable{_Coords}
    transform :: Evaluable{_Transform}
    coords :: Evaluable{_Coords}
    ndims :: Int
    eltype :: DataType
end

ApplyTrans(transform, coords, ndims) = ApplyTrans(transform, coords, ndims, Float64)
arguments(self::ApplyTrans) = Evaluable[self.transform, self.coords]
Base.eltype(self::ApplyTrans) = self.eltype
Base.ndims(self::ApplyTrans) = self.ndims

codegen(self::ApplyTrans) = __ApplyTrans()
struct __ApplyTrans end
@inline function (::__ApplyTrans)(_, trans, point)
    (point, grad) = trans(point.point, point.grad)
    (point=point, grad=grad)
end


"""
    GetProperty(arg, name)

Evaluable accessing a field of *arg* named *name*.
"""
struct GetProperty{T} <: Evaluable{T}
    arg :: Evaluable
    name :: Symbol

    function GetProperty(arg::Evaluable{_Coords}, name::Symbol)
        @assert name in [:point, :grad]
        new{_Array}(arg, name)
    end
end

arguments(self::GetProperty) = Evaluable[self.arg]
Base.eltype(self::GetProperty{_Array}) = eltype(self.arg)
Base.size(self::GetProperty{_Array}) = let n = ndims(self.arg)
    self.name == :grad ? (n, n) : (n,)
end

codegen(self::GetProperty) = __GetProperty{self.name}()
struct __GetProperty{V} end
@generated (::__GetProperty{V})(_, arg) where V = quote
    @_inline_meta
    arg.$V
end


"""
    Contract((args...), (indices...), target)

Compute a fully unrolled tensor contraction.
"""
struct Contract <: Evaluable{_Array}
    args :: Tuple{Vararg{Evaluable{_Array}}}
    indices :: Tuple{Vararg{Dims}}
    target :: Dims

    function Contract(args::Tuple{Vararg{Evaluable{_Array}}}, indices::Tuple{Vararg{Dims}}, target::Dims)
        @assert length(args) == length(indices)
        @assert all(ndims(arg) == length(ind) for (arg, ind) in zip(args, indices))

        dims = _sizedict(args, indices)
        for (arg, ind) in zip(args, indices)
            @assert all(size(arg, i) == dims[ind[i]] for i in 1:ndims(arg))
        end

        target_size = Tuple(dims[i] for i in target)
        new(args, indices, target)
    end
end

arguments(self::Contract) = self.args
Base.size(self::Contract) = let dims = _sizedict(self.args, self.indices)
    Tuple(dims[i] for i in self.target)
end

codegen(self::Contract) = __Contract{self.indices, self.target}(
    @MArray zeros(eltype(self), size(self)...)
)
struct __Contract{I,Ti,T}
    val :: T
    __Contract{I,Ti}(val) where {I,Ti} = new{I,Ti,typeof(val)}(val)
end
@generated function (self::__Contract{I,Ti})(_, args...) where {I,Ti}
    dims = _sizedict(args, I)
    dim_order = Dict(axis => num for (num, axis) in enumerate(keys(dims)))

    codes = Expr[]
    for indices in product((1:n for n in values(dims))...)
        inputs = [
            :(args[$i][$((indices[dim_order[ax]] for ax in ind)...)])
            for (i, ind) in enumerate(I)
        ]
        product = :(*($(inputs...)))
        target = :(self.val[$((indices[dim_order[ax]] for ax in Ti)...)])
        push!(codes, :($target += $product))
    end

    quote
        @_inline_meta
        @inbounds begin
            self.val .= zero(eltype(self.val))
            $(codes...)
        end
        self.val
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
struct Constant <: Evaluable{_Array}
    value :: AbstractArray
end

Base.eltype(self::Constant) = eltype(self.value)
Base.ndims(self::Constant) = ndims(self.value)
Base.size(self::Constant) = size(self.value)

codegen(self::Constant) = __Constant(self.value)
struct __Constant{T}
    val :: T
end
@inline (self::__Constant)(_) = self.val


"""
    GetIndex(arg, index...)

An evaluable returning a view into another array.
"""
struct GetIndex <: Evaluable{_Array}
    arg :: Evaluable{_Array}
    index :: Tuple{Vararg{Union{Colon, Int}}}

    function GetIndex(arg, index::Union{Colon, Int}...)
        @assert length(index) == ndims(arg)
        new(arg, index)
    end
end

arguments(self::GetIndex) = Evaluable[self.arg]
Base.size(self::GetIndex) = Tuple(s for (s,i) in zip(size(self.arg), self.index) if i isa Colon)

codegen(self::GetIndex) = __GetIndex(self.index, @MArray zeros(eltype(self), size(self)...))
struct __GetIndex{I,T}
    val :: T
    __GetIndex(I, val) = new{I, typeof(val)}(val)
end
@generated (self::__GetIndex{I})(_, arg) where I = quote
    @_inline_meta
    self.val .= arg[$(I...)]
    self.val
end


"""
    Inv(arg)

An evaluable that computes the inverse of the two-dimensional argument
*arg*.
"""
struct Inv <: Evaluable{_Array}
    arg :: Evaluable{_Array}

    function Inv(arg::Evaluable)
        @assert ndims(arg) == 2
        @assert size(arg, 1) == size(arg, 2)
        @assert size(arg, 1) < 4
        new(arg)
    end
end

arguments(self::Inv) = Evaluable[self.arg]
Base.size(self::Inv) = size(self.arg)
Base.eltype(self::Inv) = let t = eltype(self.arg)
    t <: Integer ? Float64 : t
end

codegen(self::Inv) = __Inv(@MArray zeros(eltype(self), size(self)...))
struct __Inv{T}
    val :: T
end
@generated function (self::__Inv)(_, arg)
    dims = size(arg, 1)
    T = eltype(arg)
    if dims == 1
        quote
            @inbounds begin
                self.val[1,1] = $(one(T)) / arg[1,1]
            end
            self.val
        end
    elseif dims == 2
        quote
            @inbounds begin
                self.val[1,1] = arg[2,2]
                self.val[2,2] = arg[1,1]
                self.val[1,2] = -arg[1,2]
                self.val[2,1] = -arg[2,1]
                self.val ./= (arg[1,1] * arg[2,2] - arg[1,2] * arg[2,1])
            end
            self.val
        end
    elseif dims == 3
        quote
            @inbounds begin
                self.val[1,1] = arg[2,2] * arg[3,3] - arg[2,3] * arg[3,2]
                self.val[2,1] = arg[2,3] * arg[3,1] - arg[2,1] * arg[3,3]
                self.val[3,1] = arg[2,1] * arg[3,2] - arg[2,2] * arg[3,1]
                self.val[1,2] = arg[1,3] * arg[3,2] - arg[1,2] * arg[3,3]
                self.val[2,2] = arg[1,1] * arg[3,3] - arg[1,3] * arg[3,1]
                self.val[3,2] = arg[1,2] * arg[3,1] - arg[1,1] * arg[3,2]
                self.val[1,3] = arg[1,2] * arg[2,3] - arg[1,3] * arg[2,2]
                self.val[2,3] = arg[1,3] * arg[2,1] - arg[1,1] * arg[2,3]
                self.val[3,3] = arg[1,1] * arg[2,2] - arg[1,2] * arg[2,1]
                self.val ./= (
                    arg[1,1] * self.val[1,1] +
                    arg[1,2] * self.val[2,1] +
                    arg[1,3] * self.val[3,1]
                )
            end
            self.val
        end
    end
end


"""
    Monomials(arg, degree, padding=0)

Computes all monomials of *arg* up to *degree*, with *padding* leading
zeros, yielding an array of size

    (size(arg)..., padding + degree + 1).
"""
struct Monomials <: Evaluable{_Array}
    arg :: Evaluable{_Array}
    degree :: Int
    padding :: Int
end

Monomials(arg, degree) = Monomials(arg, degree, 0)

arguments(self::Monomials) = Evaluable[self.arg]
Base.size(self::Monomials) = (size(self.arg)..., self.padding + self.degree + 1)

codegen(self::Monomials) = __Monomials{self.degree, self.padding}(@MArray zeros(eltype(self), size(self)...))
struct __Monomials{D,P,T}
    val :: T
    __Monomials{D,P}(val) where {D,P} = new{D,P,typeof(val)}(val)
end
@generated function (self::__Monomials{D,P})(_, arg) where {D,P}
    colons = [Colon() for _ in 1:ndims(arg)]
    codes = [
        :(self.val[$(colons...), $(P+i+1)] .= self.val[$(colons...), $(P+i)] .* arg)
        for i in 1:D
    ]

    quote
        @_inline_meta
        @inbounds begin
            self.val[$(colons...), 1:$P] .= $(zero(eltype(arg)))
            self.val[$(colons...), $(P+1)] .= $(one(eltype(arg)))
            $(codes...)
        end
        self.val
    end
end


"""
    Negate(arg)

Negate the argument.
"""
struct Negate <: Evaluable{_Array}
    arg :: Evaluable{_Array}
end

arguments(self::Negate) = Evaluable[self.arg]
Base.size(self::Negate) = size(self.arg)

codegen(self::Negate) = __Negate()
struct __Negate end
@inline (::__Negate)(_, arg) = -arg


"""
    Product(args...)

Elementwise product of arguments.
"""
struct Product <: Evaluable{_Array}
    args :: Vector{Evaluable{_Array}}
    dims :: Dims

    function Product(args...)
        dims = broadcast_shape(map(size, args)...)
        new(collect(Evaluable, args), dims)
    end
end

arguments(self::Product) = self.args
Base.size(self::Product) = self.dims

codegen(self::Product) = __Product(@MArray zeros(eltype(self), self.dims...))
struct __Product{T}
    val :: T
end
@generated function (self::__Product{T})(_, args...) where T
    argcodes = [:(args[$i]) for i in 1:length(args)]
    quote
        self.val .= .*($(argcodes...))
        self.val
    end
end


"""
    Reshape(arg, size...)

Reshape *arg* to a new size.
"""
struct Reshape <: Evaluable{_Array}
    arg :: Evaluable{_Array}
    shape :: Dims

    function Reshape(arg, newsize...)
        newsize = collect(Any, newsize)
        if (colon_index = findfirst(==(:), newsize)) != nothing
            in_length = prod(size(arg))
            out_length = all(==(:), newsize) ? 1 : prod(filter(!(==(:)), newsize))
            @assert in_length % out_length == 0
            newsize[colon_index] = div(in_length, out_length)
        end
        @assert all(k != (:) for k in newsize)
        new(arg, Tuple(newsize))
    end
end

arguments(self::Reshape) = Evaluable[self.arg]
Base.size(self::Reshape) = self.shape

codegen(self::Reshape) = __Reshape(@MArray zeros(eltype(self), size(self)...))
struct __Reshape{S,T}
    val :: T
    __Reshape(val) = new{size(val), typeof(val)}(val)
end
@generated function (self::__Reshape{S})(_, arg) where S
    quote
        @_inline_meta
        self.val .= reshape(arg, $(S...))
        self.val
    end
end


"""
    Sum(args...)

Elementwise sum of arguments.
"""
struct Sum <: Evaluable{_Array}
    args :: Vector{Evaluable{_Array}}
    dims :: Dims

    function Sum(args...)
        dims = broadcast_shape(map(size, args)...)
        new(collect(Evaluable, args), dims)
    end
end

arguments(self::Sum) = self.args
Base.size(self::Sum) = self.dims

codegen(self::Sum) = __Sum(@MArray zeros(eltype(self), self.dims...))
struct __Sum{T}
    val :: T
end
@generated function (self::__Sum{T})(_, args...) where T
    argcodes = [:(args[$i]) for i in 1:length(args)]
    quote
        self.val .= .+($(argcodes...))
        self.val
    end
end


"""
    Zeros(T=Float64, dims...)

Return a constant zero array of the given size and type.
"""
struct Zeros <: Evaluable{_Array}
    dims :: Dims
    eltype :: DataType
    Zeros(eltype::Type, dims::Int...) = new(dims, eltype)
end

Zeros(dims::Int...) = Zeros(Float64, dims...)
Base.eltype(self::Zeros) = self.eltype
Base.size(self::Zeros) = self.dims

codegen(self::Zeros) = __Zeros(@MArray zeros(self.eltype, self.dims...))
struct __Zeros{T}
    val :: T
    __Zeros(val) = new{typeof(val)}(val)
end
@inline (self::__Zeros)(_) = self.val
