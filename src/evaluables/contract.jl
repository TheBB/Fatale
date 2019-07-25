# A contraction is represented in terms of:
# - A vector of evaluables
# - A vector of integer vectors, each corresponding to the dimensions
#   of an evaluable
# - An integer vector, corresponding to the dimensions of the output


"""
    Contract([args...], [indices...], target)

Compute a fully unrolled tensor contraction.
"""
struct Contract <: ArrayEvaluable
    args :: Vector{Evaluable}
    indices :: Vector{Vector{Int}}
    target :: Vector{Int}

    function Contract(args, indices, target)
        @assert length(args) == length(indices)
        @assert all(ndims(arg) == length(ind) for (arg, ind) in zip(args, indices))

        dims = _sizedict(args, indices)
        for (arg, ind) in zip(args, indices)
            @assert all(size(arg, i) == dims[ind[i]] for i in 1:ndims(arg))
        end

        new(args, indices, target)
    end
end

arguments(self::Contract) = self.args
Base.size(self::Contract) = _contract_size(self.args, self.indices, self.target)

function codegen(self::Contract)
    inds = Tuple(map(Tuple, self.indices))
    target = Tuple(self.target)
    storage = @MArray zeros(eltype(self), size(self)...)
    __Contract{inds, target}(storage)
end

struct __Contract{I,Ti,T}
    val :: T
    __Contract{I,Ti}(val::T) where {I,Ti,T} = new{I,Ti,T}(val)
end

@generated function (self::__Contract{I,Ti})(args...) where {I,Ti}
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
        @inbounds begin
            self.val .= zero(eltype(self.val))
            $(codes...)
        end
        SArray(self.val)
    end
end


# Apply a contraction at 'compile time'
function _do_contract(left, right, l, r, t)
    newsize = _contract_size((left, right), (l, r), t)
    newtype = promote_type(eltype(left), eltype(right))
    ret = @MArray zeros(newtype, newsize...)
    func = __Contract{(Tuple(l), Tuple(r)), Tuple(t)}(ret)
    func(left, right)
end

# Fold an evaluable into a contraction as a new argument on the right
function _fold_contract(left::Contract, right::Evaluable, l, r, t)
    rename = MemoizedMap(Dict(zip(l, left.target)))
    new_args = Evaluable[left.args..., right]
    new_indices = Vector{Int}[left.indices..., map(rename, r)]
    new_target = map(rename, t)
    Contract(new_args, new_indices, new_target)
end

# Fold all the arguments of the right contraction into the left contraction
function _fold_contract(left::Contract, right::Contract, l, r, t)
    r_to_m = MemoizedMap(Dict(zip(right.target, r)))
    m_to_l = MemoizedMap(Dict(zip(l, left.target)))

    new_rinds = [map(m_to_l ∘ r_to_m, ri) for ri in right.indices]
    new_target = map(m_to_l, t)

    new_args = Evaluable[left.args..., right.args...]
    new_indices = Vector{Int}[left.indices..., new_rinds...]
    Contract(new_args, new_indices, new_target)
end

# Collapse the constants in positions 1 and j to a new constant on the left
function _collapse_constants!(contract, j)
    mask = setdiff(1:length(contract.args), (1,j))
    other_inds = Set(flatten(contract.indices[mask])) ∪ Set(contract.target)
    newtarget = [k for k in (contract.indices[1]..., contract.indices[j]...) if k in other_inds]
    newarg = convert(Evaluable, _do_contract(valueof(contract.args[1]), valueof(contract.args[j]), contract.indices[1], contract.indices[j], newtarget))

    contract.args[1] = newarg
    contract.indices[1] = newtarget
    deleteat!(contract.args, j)
    deleteat!(contract.indices, j)
    contract
end

# A dictionary mapping axis IDs to sizes
_sizedict(args, inds) = OrderedDict(flatten(
    (k => v for (k, v) in zip(ind, size(arg)))
    for (arg, ind) in zip(args, inds)
))

# Compute the expected size of a contraction
_contract_size(args, indices, target) = let dims = _sizedict(args, indices)
    Tuple(dims[i] for i in target)
end


# Outer constructors with purpose:
# - Contractions of contractions become bigger contractions
# - Constants accumulate in the leftmost argument, if any
# - Contraction involving zero becomes zero

# Basic conversion of arguments
Contract(left::Evaluable, right::Evaluable, l, r, t) = Contract([left, right], [l, r], t)
Contract(left::Evaluable, right, l, r, t) = Contract([left, convert(Evaluable, right)], [l, r], t)
Contract(left, right::Evaluable, l, r, t) = Contract([convert(Evaluable, left), right], [l, r], t)

# Accumulate constants to the right in new contractions
Contract(left::Evaluable, right::AbstractConstant, l, r, t) = Contract(right, left, r, l, t)

# Immediately compute contractions involving two constants
Contract(left::AbstractConstant, right::AbstractConstant, l, r, t) =
    convert(Evaluable, _do_contract(valueof(left), valueof(right), l, r, t))

# Contractions involving contractions become bigger contractions
Contract(left::Contract, right::Evaluable, l, r, t) = _fold_contract(left, right, l, r, t)
Contract(left::Evaluable, right::Contract, l, r, t) = _fold_contract(right, left, r, l, t)

# Combine constants when folding a contraction and a constant
Contract(left::AbstractConstant, right::Contract, l, r, t) = Contract(right, left, r, l, t)
function Contract(left::Contract, right::AbstractConstant, l, r, t)
    contract = _fold_contract(left, right, l, r, t)
    if contract.args[1] isa AbstractConstant
        # rgs, inds = _collapse_constants(args, inds, target, length(args))
        _collapse_constants!(contract, length(contract.args))
    end
    contract
end

# Combine constants when folding two contractions
function Contract(left::Contract, right::Contract, l, r, t)
    k = length(left.args) + 1
    contract = _fold_contract(left, right, l, r, t)
    if left.args[1] isa AbstractConstant && right.args[1] isa AbstractConstant
        _collapse_constants!(contract, k)
        # args, inds = _collapse_constants(args, inds, target, 1, k)
    elseif right.args[1] isa AbstractConstant
        contract.args[1], contract.args[k] = contract.args[k], contract.args[1]
        contract.indices[1], contract.indices[k] = contract.indices[k], contract.indices[1]
    end
    contract
end

# Contraction involving zero becomes zero
Contract(left::Evaluable, right::Zeros, l, r, t) = Contract(right, left, r, l, t)
function Contract(left::Zeros, right::Evaluable, l, r, t)
    newsize = _contract_size([left, right], [l, t], t)
    newtype = promote_type(eltype(left), eltype(right))
    Zeros(newtype, newsize...)
end