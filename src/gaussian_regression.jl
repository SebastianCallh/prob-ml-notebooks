using Random, Distributions, LinearAlgebra, Parameters, Plots
include("utils.jl")

@with_kw struct GaussianRegression{F,T <: Real}
    X::Matrix{T}
    y::Vector{T}
    μ::Vector{T}
    Σ::Matrix{T}
    σ::T
    ϕ::F
end

function prior_w(reg::GaussianRegression)
    @unpack μ, Σ = reg
    MvNormal(μ, Σ)
end

function posterior_w(reg::GaussianRegression)
    @unpack X, y, μ, Σ, σ, ϕ = reg
    ϕX = ϕ(X)
    invΣ = inv(Σ)
    A = Symmetric(invΣ + σ^(-2) * ϕX * ϕX')
    G = cholesky(A)
    A = (G \ I(size(A, 1)))'
    μₙ = A * (invΣ * μ + σ^(-2) * ϕX * y)
    Σₙ = Symmetric(posdef!(A))
    MvNormal(μₙ, Σₙ)
end

function prior_f(reg::GaussianRegression, x)
    @unpack μ, Σ, ϕ = reg
    ϕₓ = ϕ(x)
    μₙ = ϕₓ' * μ
    Σₙ = Symmetric(posdef!(ϕₓ' * Σ * ϕₓ))
    MvNormal(μₙ, Σₙ)
end

"""Here we use the alterantive posterior identity since K < N."""
function posterior_f(reg::GaussianRegression, x)
    @unpack X, y, μ, Σ, σ, ϕ = reg
    ϕX = ϕ(X)
    ϕx = ϕ(x)
    invΣ = inv(Σ)
    A = Symmetric(invΣ + σ^(-2) * ϕX * ϕX')
    G = cholesky(A)
    A = (G \ ϕx)'
    μₙ = A * (invΣ * μ + σ^(-2) * ϕX * y)
    Σₙ = Symmetric(posdef!(A * ϕx))
    MvNormal(μₙ, Σₙ)
end

"""The evidence term is the marginal likelihood and
the optimisation target for type-II maximum likelihood."""
function log_evidence(reg::GaussianRegression)
    @unpack X, y, μ, Σ, σ, ϕ = reg
    N = size(X, 2)
    Λ = σ^2 .* I(N)
    N = length(X)
    ϕX = ϕ(X)
    κXX = ϕX' * Σ * ϕX
    r = y .- ϕX' * μ
    -0.5 * r' * inv(κXX + Λ) * r - log(det(κXX + Λ)) + 0.5 * N * log(2π)
end

function plot_features(reg::GaussianRegression, xx)
    @unpack X, y = reg
    pf₀ = prior_f(reg, xx)
    fs = rand(pf₀, 5)
    prior_plt = plot(
        xx', pf₀.μ, ribbon=2 * sqrt.(diag(pf₀.Σ)),
        title="Prior predictive",
        xlabel="x",
        ylabel="f(x)",
        legend=:topleft,
        label="p(f)"
    )
    for (i, f) in enumerate(eachcol(fs))
        plot!(
            prior_plt, xx', f, color=3,
            label=i == 1 ? "Prior sample" : nothing
        )
    end

    scatter!(prior_plt, X', y, label="Observations", color=1)

    pfₙ = posterior_f(reg, xx)
    posterior_plt  = plot(
        xx', pfₙ.μ, ribbon=2 * sqrt.(diag(pfₙ.Σ)),
        color=2,
        label="p(f | X, y)", legend=:topleft
    )
    scatter!(
        posterior_plt, X', y, label="Observations",
        color=1,
        title="Posterior predictive",
        xlabel="x", ylabel="f(x)"
    )

    plot(prior_plt, posterior_plt, layout=(2, 1), size=(600, 600))
end
