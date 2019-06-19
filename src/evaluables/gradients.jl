function grad(self::Evaluable, geom::Evaluable)
    @assert ndims(geom) == 1
    dims = size(geom, 1)
    grad(self, dims) * inv(grad(geom, dims))
end

function grad(self::GetProperty{:point}, d::Int)
    @assert self.arg isa ApplyTrans
    @assert d == size(self, 1)
    self.arg.grad
end

function grad(self::GetProperty{:grad}, d::Int)
    @assert self.arg isa ApplyTrans
    @assert d == size(self, 1) == size(self, 2)
    Zeros(eltype(self), size(self)..., d)
end

function grad(self::Contract{Inds, Tinds}, d::Int) where {Inds, Tinds}
    inds = collect(Tuple(I.parameters) for I in Inds.parameters)
    tinds = Tuple(Tinds.parameters)
    next = max(maximum(tinds), (maximum(ind) for ind in inds)...)

    terms = Evaluable[]
    for (i, arg) in enumerate(self.args)
        grad_arg = grad(arg, d)
        push!(terms, Contract(
            (self.args[1:i-1]..., grad_arg, self.args[i+1:end]...),
            (inds[1:i-1]..., (inds[i]..., next), inds[i+1:end]...),
            (tinds..., next)
        ))
    end

    Sum(terms...)
end

grad(self::Inv, d::Int) = -Contract(
    (self, grad(self.arg, d), self),
    ((1,2), (2,3,5), (3,4)), (1,4,5)
)
