function svm_objective(w, Xp, y, coeff, full_function)
    n, _ = size(Xp)
    soft_margin = map(x -> max(0, x), ones(length(y)) - y .* (Xp * w))
    if full_function
        return sum(soft_margin) / n + coeff / 2 * LinearAlgebra.norm(w)^2
    end
    return sum(soft_margin) / n
end

function svm_subgradient(w, Xp, y, coeff, full_function)
    n, m = size(Xp)
    soft_margin = map(x -> max(0, x), ones(length(y)) - y .* (Xp * w))
    result = zeros(length(w))
    for i = 1:length(soft_margin)
        if soft_margin[i] > 0.0
            result += -y[i] * Xp[i, :] / n
        end
    end
    if full_function
        return result + coeff * w
    end
    return result
end

function get_sliding_min(arr)
    best_val_so_far = Inf
    arr_best_val_so_far = ones(length(arr)).+Inf
    for i in 1:length(arr)
        arr_best_val_so_far[i] = min(best_val_so_far,arr[i])
        best_val_so_far = arr_best_val_so_far[i]
    end 
    return arr_best_val_so_far
end