function Base.getproperty(self::Evaluable{T}, v::Symbol) where T<:NamedTuple
    index = findfirst(x->x==v, T.parameters[1])
    index == nothing && return getfield(self, v)
    rtype = T.parameters[2].parameters[index]
    GetProperty{v, rtype}(self)
end


# ==============================================================================
# Convenience constructors

localpoint(n) = LocalCoords(n).point
localgrad(n) = LocalCoords(n).grad
globalpoint(n) = GlobalCoords(n).point
globalgrad(n) = GlobalCoords(n).grad
