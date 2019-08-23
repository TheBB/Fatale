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
eltype(self::Inflate) = eltype(self.arg)
size(self::Inflate) = Tuple(
    i == self.axis ? self.newsize : size(self.arg, i)
    for i in 1:ndims(self.arg)
)

blocks(self::Inflate) = ((
    indices = (blk.indices[1:self.axis-1]..., self.indices, blk.indices[self.axis+1:end]...),
    data = blk.data,
) for blk in blocks(self.arg))


"""
    ElementIntegral(arg)

Integrate *arg* over the current element.
"""
struct ElementIntegral <: ArrayEvaluable
    arg :: ArrayEvaluable
end

arguments(self::ElementIntegral) = Evaluable[self.arg, ElementData{_Transform}(:loctrans), EvalArg{_Any}(:quadrule)]
size(self::ElementIntegral) = size(self.arg)
eltype(self::ElementIntegral) = let t = eltype(self.arg)
    t <: Integer ? Float64 : t
end

codegen(self::ElementIntegral) = Cpl.ElementIntegral{sarray(self)}()


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
eltype(self::GetIndex) = eltype(self.arg)

function size(self::GetIndex)
    sz = collect(Int, size(self.arg))
    ret = Int[]
    for ix in self.index
        push!(ret, size(ix)...)
        sz = sz[_consumedims(ix)+1:end]
    end
    Tuple(ret)
end

codegen(self::GetIndex) = Cpl.GetIndex()


"""
    Gradient(arg)

Take the gradient of `arg` with respect to input parameters.
"""
struct Gradient <: ArrayEvaluable
    arg :: ArrayEvaluable
    d :: Int
end

arguments(self::Gradient) = Evaluable[self.arg]
size(self::Gradient) = (self.d, size(self.arg)...)
eltype(self::Gradient) = eltype(self.arg)

codegen(self::Gradient) = Cpl.Gradient{size(self)}()


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
size(self::Inv) = size(self.arg)
eltype(self::Inv) = let t = eltype(self.arg)
    t <: Integer ? Float64 : t
end

codegen(self::Inv) = Cpl.Inv()


"""
    Monomials(arg, degree, padding=0)

Computes all monomials of *arg* up to *degree*, with *padding* leading
zeros, yielding an array of size

    (padding + degree + 1, size(arg)...).
"""
struct Monomials <: ArrayEvaluable
    arg :: ArrayEvaluable
    degree :: Int
    padding :: Int
end

Monomials(arg, degree) = Monomials(arg, degree, 0)

arguments(self::Monomials) = Evaluable[self.arg]
size(self::Monomials) = (self.padding + self.degree + 1, size(self.arg)...)

codegen(self::Monomials) = Cpl.Monomials{self.degree, self.padding, size(self)}()


"""
    Negate(arg)

Negate the argument.
"""
struct Negate <: ArrayEvaluable
    arg :: ArrayEvaluable
end

arguments(self::Negate) = Evaluable[self.arg]
size(self::Negate) = size(self.arg)

codegen(self::Negate) = Cpl.Negate()


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
size(self::PermuteDims) = Tuple(size(self.arg, i) for i in self.perm)

codegen(self::PermuteDims) = Cpl.PermuteDims{self.perm}()


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
size(self::Power) = size(self.arg)

codegen(self::Power) = Cpl.Power{self.exp}()


"""
    Reciprocal(arg)

Compute the elementwise reciprocal of *arg*.
"""
struct Reciprocal <: ArrayEvaluable
    arg :: ArrayEvaluable
end

arguments(self::Reciprocal) = Evaluable[self.arg]
size(self::Reciprocal) = size(self.arg)
eltype(self::Reciprocal) = let t = eltype(self.arg)
    t <: Integer ? Float64 : t
end

codegen(self::Reciprocal) = Cpl.Reciprocal()


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
size(self::Reshape) = self.shape

codegen(self::Reshape) = Cpl.Reshape{size(self)}()


"""
    Sqrt(arg)

Elementwise square root.
"""
struct Sqrt <: ArrayEvaluable
    arg :: ArrayEvaluable
end

arguments(self::Sqrt) = Evaluable[self.arg]
size(self::Sqrt) = size(self.arg)

codegen(self::Sqrt) = Cpl.Sqrt()


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
function size(self::Sum)
    if self.collapse
        Tuple(k for (i, k) in enumerate(size(self.arg)) if !(i in self.dims))
    else
        Tuple(i in self.dims ? 1 : k for (i, k) in enumerate(size(self.arg)))
    end
end

codegen(self::Sum) = Cpl.Sum{self.dims, size(self)}()
