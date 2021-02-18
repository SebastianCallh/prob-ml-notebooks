
"""Eigenvalues can be slightly negative after computing them."""
posdef!(Σ; ϵ=1e-6) = begin
    for i in 1:size(Σ, 1)
        Σ[i,i] += ϵ
    end
    Σ
end
