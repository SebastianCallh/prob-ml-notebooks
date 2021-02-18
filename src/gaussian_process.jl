using Distributions, LinearAlgebra, Parameters

include("utils.jl")
constant(y) = x -> ones(size(x, 2)) .* y

σ = 5.
linear(x₀) = (x, x′) -> (x .- x₀)' * (x′ .- x₀)
rbf(ℓ) = (x, x′) -> σ .* exp.(-(x' .- x′).^2) ./ 2ℓ
periodic(ℓ, ρ) = (x, x′) -> σ .* exp.(-2 * sin.(π .* (x' .- x′) ./ ρ).^2 ./ ℓ^2)
rq(ℓ, α) = (x, x′) -> σ .* (1 .+ (x' .- x′).^2).^(-α) ./ (2α * ℓ^2)

@with_kw struct GaussianProcess{M,K,T <: Number}
    X::AbstractArray{T}
    y::AbstractArray{T}
    m::M
    k::K
    σ::T
end

function prior(gp::GaussianProcess, x)
    @unpack m, k = gp
    μ = m(x)
    Σ = Symmetric(posdef!(k(x, x)))
    MvNormal(μ, Σ)
end

function posterior(gp::GaussianProcess, x)
    @unpack X, y, m, k, σ = gp
    @show size.((x, X))
    κxX = k(x, X)
    κXX = k(X, X)
    κxx = k(x, x)
    r = y - m(X)
    A = κXX + σ^2 * I(size(X, 2))
    G = cholesky(Symmetric(posdef!(A)))
    A = (G \ κxX')'
    μ = m(x) + A * r
    Σ = Symmetric(posdef!(κxx - A * κxX'))
    MvNormal(μ, Σ)
end

function log_evidence(gp::GaussianProcess, X, y)
    @unpack X, m, k, σ = gp
    Λ = σ.^2 * I(size(X, 2))
    κXX = k(X, X)
    r = y .- m(X)
    r' * inv(κXX + Λ) * r + log(det(κXX + Λ))
end
