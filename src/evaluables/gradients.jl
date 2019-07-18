function grad(self::Evaluable, geom::Evaluable)
    @assert ndims(geom) == 1
    dims = size(geom, 1)
    grad(self, dims) * inv(grad(geom, dims))
end

function grad(self::GetProperty{_Array}, d::Int)
    @assert self.name in [:point, :grad]
    @assert self.arg isa ApplyTrans
    @assert d == size(self, 1)

    if self.name == :point
        GetProperty(self.arg, :grad)
    else
        Zeros(eltype(self), size(self)..., d)
    end
end

function grad(self::Add, d::Int)
    maxdims = maximum(ndims(term) for term in self.args)
    terms = [grad(flushleft(factor, maxdims), d) for factor in self.args]
    Add(terms...)
end

function grad(self::Contract, d::Int)
    next = max(maximum(self.target), (maximum(ind) for ind in self.indices)...) + 1

    terms = Evaluable[]
    for (i, arg) in enumerate(self.args)
        grad_arg = grad(arg, d)
        if !(grad_arg isa Zeros)
            push!(terms, Contract(
                (self.args[1:i-1]..., grad_arg, self.args[i+1:end]...),
                (self.indices[1:i-1]..., (self.indices[i]..., next), self.indices[i+1:end]...),
                (self.target..., next)
            ))
        end
    end
    Add(terms...)
end

function grad(self::Monomials, d::Int) where {D, P}
    newmono = Monomials(self.arg, self.degree-1, self.padding+1)
    scale = flushright(Constant(SVector(zeros(Int, self.padding+1)..., 1:self.degree...)), newmono)
    chain = grad(insertaxis(self.arg; right=1), d)
    insertaxis(newmono .* scale; right=1) .* chain
end

function grad(self::Multiply, d::Int)
    maxdims = maximum(ndims(factor) for factor in self.args)
    reshaped = [insertaxis(flushleft(factor, maxdims); right=1) for factor in self.args]
    terms = Evaluable[]
    for (i, factor) in enumerate(self.args)
        term = Multiply(reshaped[1:i-1]..., grad(flushleft(factor, maxdims), d), reshaped[i+1:end]...)
        push!(terms, term)
    end
    Add(terms...)
end

grad(self::Constant, d::Int) = Zeros(eltype(self), size(self)..., d)
grad(self::GetIndex, d::Int) where I = GetIndex(grad(self.arg, d), self.index..., :)
grad(self::Inv, d::Int) = -Contract(self * grad(self.arg, d), self, (1, 2, 3), (2, 4), (1, 4, 3))
grad(self::Reshape, d::Int) = reshape(grad(self.arg, d), size(self)..., d)
grad(self::PermuteDims, d::Int) = permutedims(grad(self.arg, d), (self.perm..., ndims(self) + 1))
