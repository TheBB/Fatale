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

    function Contract(args::Vector{<:Evaluable}, indices::Vector{Vector{Int}}, target::Vector{Int})
        @assert length(args) == length(indices)
        @assert all(ndims(arg) == length(ind) for (arg, ind) in zip(args, indices))

        dims = Cpl.contract_sizedict(args, indices)
        for (arg, ind) in zip(args, indices)
            @assert all(size(arg, i) == dims[ind[i]] for i in 1:ndims(arg))
        end

        new(args, indices, target)
    end
end

arguments(self::Contract) = self.args
size(self::Contract) = _contract_size(self.args, self.indices, self.target)

function codegen(self::Contract)
    inds = Tuple(map(Tuple, self.indices))
    target = Tuple(self.target)
    Cpl.Contract{inds, target, size(self)}()
end


# Apply a contraction at 'compile time'
function _do_contract(left, right, l, r, t)
    newsize = _contract_size([left, right], [l, r], t)
    func = Cpl.Contract{(Tuple(l), Tuple(r)), Tuple(t), newsize}()
    func(left, right)
end

function _do_contract(left, l, t)
    newsize = _contract_size([left], [l], t)
    func = Cpl.Contract{(Tuple(l),), Tuple(t), newsize}()
    func(left)
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
    newarg = convert(Evaluable, _do_contract(
        valueof(contract.args[1]), valueof(contract.args[j]),
        contract.indices[1], contract.indices[j], newtarget
    ))

    contract.args[1] = newarg
    contract.indices[1] = newtarget
    deleteat!(contract.args, j)
    deleteat!(contract.indices, j)
    contract
end

# Compute the expected size of a contraction
_contract_size(args, indices, target) = let dims = Cpl.contract_sizedict(args, indices)
    Tuple(dims[i] for i in target)
end


# Helper function for applying a mass contraction with optimizations
# This acts as reduce(Contract, ...) while keeping track of axis labels
function reduce_contract(args::Vector{<:Evaluable}, indices::Vector{Vector{Int}}, target::Vector{Int})
    init = (args[1], indices[1])
    rest = zip(args[2:end], indices[2:end])
    (full, fulli) = foldl(rest; init=init) do (acc, acci), (arg, argi)
        newi = vcat(acci, argi)
        new = Contract(acc, arg, acci, argi, newi)
        (new, newi)
    end
    Contract(full, fulli, target)
end


# Outer constructors with purpose:
# - Contractions of contractions become bigger contractions
# - Constants accumulate in the leftmost argument, if any
# - Contraction involving zero becomes zero

# Single-argument case for extracting diagonals etc.
Contract(left::Evaluable, l, t) = Contract([left], [l], t)

# Basic conversion of arguments
Contract(left::Evaluable, right::Evaluable, l, r, t) = Contract([left, right], [l, r], t)
Contract(left::Evaluable, right, l, r, t) = Contract([left, convert(Evaluable, right)], [l, r], t)
Contract(left, right::Evaluable, l, r, t) = Contract([convert(Evaluable, left), right], [l, r], t)

# Accumulate constants to the right in new contractions
Contract(left::Evaluable, right::AbstractConstant, l, r, t) = Contract(right, left, r, l, t)

# Immediately compute contractions involving two constants
Contract(left::AbstractConstant, right::AbstractConstant, l, r, t) =
    convert(Evaluable, _do_contract(valueof(left), valueof(right), l, r, t))
Contract(left::AbstractConstant, l, t) = convert(Evaluable, _do_contract(valueof(left), l, t))

# Contractions involving contractions become bigger contractions
Contract(left::Contract, right::Evaluable, l, r, t) = _fold_contract(left, right, l, r, t)
Contract(left::Evaluable, right::Contract, l, r, t) = _fold_contract(right, left, r, l, t)
function Contract(left::Contract, l, t)
    rename = Dict(zip(l, left.target))
    newtarget = [rename[label] for label in t]
    Contract(left.args, left.indices, newtarget)
end

# Combine constants when folding a contraction and a constant
Contract(left::AbstractConstant, right::Contract, l, r, t) = Contract(right, left, r, l, t)
function Contract(left::Contract, right::AbstractConstant, l, r, t)
    contract = _fold_contract(left, right, l, r, t)
    if contract.args[1] isa AbstractConstant
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
    elseif right.args[1] isa AbstractConstant
        swap!(contract.args, 1, k)
        swap!(contract.indices, 1, k)
    end
    contract
end

# Contraction involving zero becomes zero
Contract(left::Zeros, l, t) = Zeros(eltype(left), _contract_size([left], [l], t)...)
Contract(left::Evaluable, right::Zeros, l, r, t) = Contract(right, left, r, l, t)
function Contract(left::Zeros, right::Evaluable, l, r, t)
    newsize = _contract_size([left, right], [l, r], t)
    newtype = promote_type(eltype(left), eltype(right))
    Zeros(newtype, newsize...)
end

# Contraction must commute with inflation
Contract(left::Evaluable, right::Inflate, l, r, t) = Contract(right, left, r, l, t)

function Contract(left::Inflate, right::Evaluable, l, r, t)
    infaxis = left.axis
    infid = l[infaxis]
    indices = left.indices

    # Apply a getindex operation on the right argument
    # so that it matches the nonzero entries on the left
    newright = getindex(right, (ri == infid ? indices : Colon() for ri in r)...)

    # Apply the contraction as specified
    ret = Contract(left.arg, newright, l, r, t)

    # Inflate the axes that correspond to the originally inflated one, if any
    new_infaxes = findall(==(infid), t)
    for infaxis in new_infaxes
        ret = Inflate(ret, indices, left.newsize, infaxis)
    end

    ret
end

function Contract(left::Inflate, l, t)
    # Apply the contraction as specified
    ret = Contract(left.arg, l, t)

    # Inflate the axes that correspond to the originally inflated one, if any
    new_infaxes = findall(==(l[left.infaxis]), t)
    for infaxis in new_infaxes
        ret = Inflate(ret, left.indices, left.newsize, infaxis)
    end

    ret
end
