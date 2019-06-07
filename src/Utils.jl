module Utils

export outer


function outer(factors...)
    .*((reshape(f, fill(1, k-1)..., :) for (k, f) in enumerate(factors))...)
end

end
