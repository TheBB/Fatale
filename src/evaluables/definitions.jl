"""
    Inflate(arg, indices, newsize, axis)

Inflate the dimension number *axis* of *arg* to have size *newsize*,
using *indices* for placing.
"""
struct Inflate <: ArrayEvaluable
    arg :: ArrayEvaluable
    indices :: ArrayEvaluable
    newsize :: Int
    axis :: Int

    function Inflate(arg, indices, newsize, axis)
        @assert 1 <= axis <= ndims(arg)
        @assert ndims(indices) == 1
        @assert size(indices, 1) == size(arg, axis)
        new(arg, indices, newsize, axis)
    end
end

function Inflate(arg, indices, newsize)
    @assert ndims(arg) == 1
    Inflate(arg, indices, newsize, 1)
end

arguments(self::Inflate) = Evaluable[self.arg, self.indices]
Base.eltype(self::Inflate) = eltype(self.arg)
Base.size(self::Inflate) = Tuple(
    i == self.axis ? self.newsize : size(self.arg, i)
    for i in 1:ndims(self.arg)
)

blocks(self::Inflate) = ((
    indices = (blk.indices[1:self.axis-1]..., self.indices, blk.indices[self.axis+1:end]...),
    data = blk.data,
) for blk in blocks(self.arg))


"""
    GetIndex(arg, index...)

An evaluable returning a view into another array.
"""
struct GetIndex <: ArrayEvaluable
    arg :: ArrayEvaluable
    index :: Tuple

    function GetIndex(arg, index...)
        cleaned = map(asevaluable, index, size(arg))
        @assert sum(_consumedims, cleaned) == ndims(arg)
        new(arg, cleaned)
    end
end

_consumedims(s::ArrayEvaluable) = let t = eltype(s)
    t <: Integer && return 1
    t <: CartesianIndex && return length(t)
    @assert false
end

arguments(self::GetIndex) = Evaluable[
    self.arg,
    Iterators.filter(x->x isa ArrayEvaluable, self.index)...
]
Base.eltype(self::GetIndex) = eltype(self.arg)

function Base.size(self::GetIndex)
    sz = collect(Int, size(self.arg))
    ret = Int[]
    for ix in self.index
        push!(ret, size(ix)...)
        sz = sz[_consumedims(ix)+1:end]
    end
    Tuple(ret)
end

codegen(self::GetIndex) = __GetIndex()
struct __GetIndex end
@generated function (self::__GetIndex)(arg, indices...)
    inds = Expr[:(indices[$i]) for i in 1:length(indices)]
    quote
        @_inline_meta
        @inbounds arg[$(inds...)]
    end
end


"""
    Inv(arg)

An evaluable that computes the inverse of the two-dimensional argument
*arg*.
"""
struct Inv <: ArrayEvaluable
    arg :: ArrayEvaluable

    function Inv(arg::Evaluable)
        @assert ndims(arg) == 2
        @assert size(arg, 1) == size(arg, 2)
        @assert size(arg, 1) <= 14
        new(arg)
    end
end

arguments(self::Inv) = Evaluable[self.arg]
Base.size(self::Inv) = size(self.arg)
Base.eltype(self::Inv) = let t = eltype(self.arg)
    t <: Integer ? Float64 : t
end

codegen(self::Inv) = __Inv()
struct __Inv end
@inline (::__Inv)(arg) = inv(arg)


"""
    Monomials(arg, degree, padding=0)

Computes all monomials of *arg* up to *degree*, with *padding* leading
zeros, yielding an array of size

    (size(arg)..., padding + degree + 1).
"""
struct Monomials <: ArrayEvaluable
    arg :: ArrayEvaluable
    degree :: Int
    padding :: Int
end

Monomials(arg, degree) = Monomials(arg, degree, 0)

arguments(self::Monomials) = Evaluable[self.arg]
Base.size(self::Monomials) = (size(self.arg)..., self.padding + self.degree + 1)

codegen(self::Monomials) = __Monomials(self.degree, self.padding, eltype(self), size(self))
struct __Monomials{D,P,T}
    val :: T
    function __Monomials(D, P, T, sz)
        val = @MArray zeros(T, sz...)
        new{D,P,typeof(val)}(val)
    end
end
@generated function (self::__Monomials{D,P})(arg) where {D,P}
    colons = [Colon() for _ in 1:ndims(arg)]
    codes = [
        :(self.val[$(colons...), $(P+i+1)] .= self.val[$(colons...), $(P+i)] .* arg)
        for i in 1:D
    ]

    quote
        @inbounds begin
            self.val[$(colons...), 1:$P] .= $(zero(eltype(arg)))
            self.val[$(colons...), $(P+1)] .= $(one(eltype(arg)))
            $(codes...)
        end
        SArray(self.val)
    end
end


"""
    Negate(arg)

Negate the argument.
"""
struct Negate <: ArrayEvaluable
    arg :: ArrayEvaluable
end

arguments(self::Negate) = Evaluable[self.arg]
Base.size(self::Negate) = size(self.arg)

codegen(self::Negate) = __Negate()
struct __Negate end
@inline (::__Negate)(arg) = -arg


"""
    PermuteDims(arg, perm)

Permute dimensions of *arg*.
"""
struct PermuteDims <: ArrayEvaluable
    arg :: ArrayEvaluable
    perm :: Dims

    function PermuteDims(arg::ArrayEvaluable, perm::Dims)
        @assert Set(perm) == Set(1:ndims(arg))
        @assert length(perm) == ndims(arg)
        new(arg, perm)
    end
end

arguments(self::PermuteDims) = Evaluable[self.arg]
Base.size(self::PermuteDims) = Tuple(size(self.arg, i) for i in self.perm)

codegen(self::PermuteDims) = __PermuteDims{self.perm}()
struct __PermuteDims{I} end
@generated function (::__PermuteDims{I})(arg) where I
    insize = size(arg)
    outsize = Tuple(size(arg, i) for i in I)

    lininds = LinearIndices(insize)
    indices = (lininds[(cind[i] for i in I)...] for cind in CartesianIndices(outsize))
    exprs = (:(arg[$i]) for i in indices)
    quote
        @_inline_meta
        SArray{Tuple{$(outsize...)}}($(exprs...))
    end
end


"""
    Power(arg, exp)

Elementwise power with fixed real exponents.
"""
struct Power <: ArrayEvaluable
    arg :: ArrayEvaluable
    exp :: Real

    function Power(arg, exp::Real)
        exp == -1 && return Reciprocal(arg)
        exp == 0 && return Constant(ones(arg))
        exp == 1/2 && return Sqrt(arg)
        exp == 1 && return arg
        new(arg, exp)
    end
end

arguments(self::Power) = Evaluable[self.arg]
Base.size(self::Power) = size(self.arg)

codegen(self::Power) = __Power{self.exp}()
struct __Power{P} end
@inline (::__Power{P})(arg::Scalar) where P = Scalar(arg[] ^ P)
@inline (::__Power{P})(arg) where P = arg .^ P


"""
    Reciprocal(arg)

Compute the elementwise reciprocal of *arg*.
"""
struct Reciprocal <: ArrayEvaluable
    arg :: ArrayEvaluable
end

arguments(self::Reciprocal) = Evaluable[self.arg]
Base.size(self::Reciprocal) = size(self.arg)

codegen(self::Reciprocal) = __Reciprocal()
struct __Reciprocal end
@inline (::__Reciprocal)(arg::Scalar) = Scalar(one(eltype(arg)) / arg[])
@inline (::__Reciprocal)(arg) = one(eltype(arg)) ./ arg


"""
    Reshape(arg, size...)

Reshape *arg* to a new size.
"""
struct Reshape <: ArrayEvaluable
    arg :: ArrayEvaluable
    shape :: Dims
    Reshape(arg, newsize...) = new(arg, Tuple(newsize))
end

arguments(self::Reshape) = Evaluable[self.arg]
Base.size(self::Reshape) = self.shape

codegen(self::Reshape) = __Reshape{size(self)}()
struct __Reshape{S} end
@generated (self::__Reshape{S})(arg) where S = quote
    @_inline_meta
    SArray{Tuple{$(S...)}}(arg)
end


"""
    Sqrt(arg)

Elementwise square root.
"""
struct Sqrt <: ArrayEvaluable
    arg :: ArrayEvaluable
end

arguments(self::Sqrt) = Evaluable[self.arg]
Base.size(self::Sqrt) = size(self.arg)

codegen(self::Sqrt) = __Sqrt()
struct __Sqrt end
@inline (::__Sqrt)(arg) = sqrt.(arg)
@inline (::__Sqrt)(arg::Scalar) = Scalar(sqrt(arg[1]))


"""
    Sum(arg, dims...)

Collapse some axes by summation.
"""
struct Sum <: ArrayEvaluable
    arg :: ArrayEvaluable
    dims :: Dims
    collapse :: Bool

    function Sum(arg::ArrayEvaluable, dims, collapse::Bool)
        if dims isa Colon
            dims = Tuple(1:ndims(arg))
        end
        @assert all(1 <= d <= ndims(arg) for d in dims)
        @assert length(Set(dims)) == length(dims)
        new(arg, dims, collapse)
    end
end

arguments(self::Sum) = Evaluable[self.arg]
function Base.size(self::Sum)
    if self.collapse
        Tuple(k for (i, k) in enumerate(size(self.arg)) if !(i in self.dims))
    else
        Tuple(i in self.dims ? 1 : k for (i, k) in enumerate(size(self.arg)))
    end
end

codegen(self::Sum) = __Sum(self.dims, size(self))
struct __Sum{D,S}
    __Sum(D,S) = new{D,S}()
end
@generated function (self::__Sum{D,S})(arg) where {D,S}
    D = collect(D)
    tempsize = Tuple(i in D ? 1 : k for (i, k) in enumerate(size(arg)))
    indexer = LinearIndices(size(arg))

    # We'd like to just call the StaticArrays implementation, but it
    # can cause allocations
    sums = Expr[]
    for i in Base.product((1:k for k in tempsize)...)
        ix = collect(i)
        exprs = Expr[]
        for px in Base.product((1:size(arg, d) for d in D)...)
            ix[D] = collect(px)
            push!(exprs, :(arg[$(indexer[ix...])]))
        end
        push!(sums, :(+($(exprs...))))
    end

    :(@inbounds SArray{Tuple{$(S...)}}($(sums...)))
end
