function grad(self::Evaluable, geom::Evaluable)
    @assert ndims(geom) == 1
    dims = size(geom, 1)
    inv(grad(geom, dims)) * grad(self, dims)
end

function grad(self::ExtractCoords, d::Int)
    @assert d == ndims(self.arg)
    @assert 0 <= self.stage <= 1
    self.stage < 1 && return ExtractCoords(self.arg, self.stage + 1)
    return Zeros(eltype(self), d, size(self)...)
end

function grad(self::Contract, d::Int)
    next = max(maximum(self.target), (maximum(ind) for ind in self.indices)...) + 1
    new_target = [next, self.target...]
    terms = map(enumerate(zip(self.args, self.indices))) do (i, (arg, ind))
        reduce_contract(
            [self.args[1:i-1]..., grad(arg, d), self.args[i+1:end]...],
            [self.indices[1:i-1]..., [next, ind...], self.indices[i+1:end]...],
            new_target,
        )
    end
    Add(terms...)
end

function grad(self::Monomials, d::Int) where {D, P}
    newmono = Monomials(self.arg, self.degree-1, self.padding+1)
    scale = SVector(zeros(Int, self.padding+1)..., 1:self.degree...)
    chain = grad(insertaxis(self.arg; left=1), d)
    insertaxis(newmono .* scale; left=1) .* chain
end

function grad(self::Multiply, d::Int)
    reshaped = [insertaxis(factor; left=1) for factor in self.args]
    terms = Evaluable[]
    for (i, factor) in enumerate(self.args)
        term = Multiply(reshaped[1:i-1]..., grad(factor, d), reshaped[i+1:end]...)
        push!(terms, term)
    end
    Add(terms...)
end

grad(self::Add, d::Int) = Add((grad(term, d) for term in self.args)...)
grad(self::Constant, d::Int) = Zeros(eltype(self), d, size(self)...)
grad(self::GetIndex, d::Int) where I = GetIndex(grad(self.arg, d), :, self.index...)
grad(self::Inflate, d::Int) = Inflate(grad(self.arg, d), self.indices, self.newsize, self.axis + 1)
grad(self::Inv, d::Int) = -Contract(grad(self.arg, d) * self, self, [1, 2, 3], [4, 2], [1, 4, 3])
grad(self::PermuteDims, d::Int) = permutedims(grad(self.arg, d), (1, map(x->x+1, self.perm)...))
grad(self::Power, d::Int) = grad(self.arg, d) .* insertaxis(self.exp .* Power(self.arg, self.ex - 1); left=1)
grad(self::Reciprocal, d::Int) = -grad(self.arg, d) ./ insertaxis(self.^2; left=1)
grad(self::Reshape, d::Int) = reshape(grad(self.arg, d), d, size(self)...)
grad(self::Sqrt, d::Int) = grad(self.arg, d) ./ insertaxis(2 .* self; left=1)
