"""
    EvalArgs()

A special Evaluable that just returns the evaluation arguments in raw
form.
"""
struct EvalArgs <: Evaluable{_Any} end
codegen(::EvalArgs) = __EvalArgs()
struct __EvalArgs end
@inline (::__EvalArgs)(arg) = arg


"""
    Funcall

A special Evaluable that represents a function call of the form

    functionname(argument, parameter)

Here, the function name is a symbol, the argument is another Evaluable
and the parameter is any object of bits type.

A funcall may represent any return type. Use one of the constructors:

    Funcall(_Array, functionname, argument, parameter, eltype, size)
    Funcall(_Coords, functionname, argument, parameter, eltype, (ndims,))
    Funcall(_Element, functionname, argument, parameter)
    Funcall(_Transform, functionname, argument, parameter)
"""
struct Funcall{T} <: Evaluable{T}
    funcname :: Symbol
    argument :: Evaluable
    parameter

    # This is a hack so that one struct can represent many different
    # output types
    eltype
    size
    ndims

    function Funcall{T}(func, arg, param; size=nothing, eltype=nothing, ndims=nothing) where T
        T <: _Coords && @assert ndims isa Int
        T <: _Array && @assert size isa Dims
        T <: Union{_Coords,_Array} && @assert eltype isa DataType
        new{T}(func, arg, param, eltype, size, ndims)
    end
end

Funcall(::Type{_Coords}, func, arg, param, eltype, (n,)) = Funcall{_Coords}(func, arg, param; eltype=eltype, ndims=n)
Funcall(::Type{_Array}, func, arg, param, eltype, size) = Funcall{_Array}(func, arg, param; eltype=eltype, size=size)
Funcall(T::Type{<:Result}, func, arg, param) = Funcall{T}(func, arg, param)

arguments(self::Funcall) = Evaluable[self.argument]
Base.size(self::Funcall{_Array}) = self.size
Base.ndims(self::Funcall{_Coords}) = self.ndims
Base.eltype(self::Funcall{<:Union{_Array,_Coords}}) = self.eltype

codegen(self::Funcall) = __Funcall{self.funcname, self.parameter}()
struct __Funcall{F,A} end
@generated function (::__Funcall{F,A})(arg) where {F,A}
    quote
        @_inline_meta
        $F(arg, :($A))
    end
end


"""
    ApplyTrans(trans, coords)

Apply `trans` to `coords`.
"""
struct ApplyTrans <: CoordsEvaluable
    transform :: Evaluable{_Transform}
    coords :: CoordsEvaluable
end

Base.eltype(self::ApplyTrans) = eltype(self.coords)
Base.ndims(self::ApplyTrans) = ndims(self.coords)
arguments(self::ApplyTrans) = Evaluable[self.transform, self.coords]

codegen(self::ApplyTrans) = __ApplyTrans()
struct __ApplyTrans end
@inline (::__ApplyTrans)(trans, point) = trans(point)


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
    Add(args...)

Elementwise sum of arguments.
"""
struct Add <: ArrayEvaluable
    args :: Vector{Evaluable}
    dims :: Dims

    function Add(args::VarTuple{ArrayEvaluable})
        length(args) == 1 && return args[1]
        # Simplification: constants should accumulate on the left
        @assert all(!(arg isa AbstractConstant) for arg in args[2:end])
        dims = broadcast_shape(map(size, args)...)
        new(collect(Evaluable, args), dims)
    end
end

arguments(self::Add) = self.args
Base.size(self::Add) = self.dims

codegen(self::Add) = __Add()
struct __Add end
@generated function (self::__Add)(args...)
    argcodes = [:(args[$i]) for i in 1:length(args)]
    quote
        .+($(argcodes...))
    end
end


"""
    Constant(v)

An evaluable returning the constant object *v*.
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
    Multiply(args...)

Elementwise product of arguments.
"""
struct Multiply <: ArrayEvaluable
    args :: Vector{Evaluable}
    dims :: Dims

    function Multiply(args::VarTuple{ArrayEvaluable})
        length(args) == 1 && return args[1]
        # Simplification: constants should accumulate on the left
        @assert all(!(arg isa AbstractConstant) for arg in args[2:end])
        dims = broadcast_shape(map(size, args)...)
        new(collect(Evaluable, args), dims)
    end
end

arguments(self::Multiply) = self.args
Base.size(self::Multiply) = self.dims

codegen(self::Multiply) = __Multiply()
struct __Multiply end
@generated function (self::__Multiply)(args...)
    argcodes = [:(args[$i]) for i in 1:length(args)]
    quote
        .*($(argcodes...))
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

    # We'd like to just call the StaticArrays implementation, but it
    # can cause allocations
    sums = Expr[]
    for i in Base.product((1:k for k in tempsize)...)
        ix = collect(i)
        exprs = Expr[]
        for px in Base.product((1:size(arg, d) for d in D)...)
            ix[D] = collect(px)
            push!(exprs, :(arg[$(ix...)]))
        end
        push!(sums, :(+($(exprs...))))
    end

    :(@inbounds SArray{Tuple{$(S...)}}($(sums...)))
end


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
struct Zeros <: ArrayEvaluable
    dims :: Dims
    eltype :: DataType
    Zeros(eltype::Type, dims::Int...) = new(dims, eltype)
end

Zeros(dims::Int...) = Zeros(Float64, dims...)
Base.eltype(self::Zeros) = self.eltype
Base.size(self::Zeros) = self.dims

codegen(self::Zeros) = __Zeros{SArray{Tuple{size(self)...}, eltype(self), ndims(self), length(self)}}()
struct __Zeros{T} end
@inline (self::__Zeros{T})() where T = zero(T)
