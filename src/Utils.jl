module Utils

export chain, outer, exterior


chain(list) = zip(list[1:end-1], list[2:end])

function outer(factors...)
    .*((reshape(f, fill(1, k-1)..., :) for (k, f) in enumerate(factors))...)
end

exterior(func; repeat=2) = outer(Iterators.repeated(func, repeat)...)

end
