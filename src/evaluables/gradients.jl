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
