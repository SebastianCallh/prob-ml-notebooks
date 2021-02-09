### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 1b3d1874-6a0f-11eb-1b72-a5e692d0c7e6
begin
	using Plots, Distributions, ConjugatePriors, PlutoUI
	plotly()
end;

# ╔═╡ 5f4db9e4-6a26-11eb-2257-97e721b8727e
md"""
# Normal Inverse-Wishart distributions as conjugate prior
In this notebook we are going to look at the Normal Inverse-Wishart (NIW) distribution and its conjugacy of the multivariate normal distribution.

### Motivation
Bayesian inference is a computationally challenging task. In the general case, it is exponentially expensive to compute posterior distributions. Fortuantely, clever mathematicians have developed several smart tricks to make inference tractable. One particularly elegant trick is that of *conjugate priors*, which makes it possible to perform inference in closed form. A distribution is a conjugate prior with respect to a specific likelihood if the posterior is of the same family. The NIW is the conjugate prior to a multivariate Gaussian with unkown mean and variance.


### Parametrisation and inference
The Normal Inverse Wishart distribution $NIW(μ₀, κ₀, Λ₀, ν₀)$ has four parameters.
* μ₀ is the prior mean vector
* Λ₀ is the prior scale matrix
* κ₀ is prior pseudocount for the mean
* ν₀ is prior pseudocount for the covariance

In this notebook we are using notation from [Conjugate Bayesian analysis of the Gaussian distribution](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf). Additinally, we use the same equation numbering to make it easy to track equations in the original source.

Given the apove parametrisation the posterior density is

\begin{equation}
\tag{250}
p(\mu, \Sigma \vert \mathcal{D}, \mu\_0, \kappa\_0, \Lambda\_0, \nu\_0) = 
NIW(\mu, \Sigma \vert \mu\_n, \kappa\_n, \Lambda\_n, \nu\_n),
\end{equation}
where

\begin{equation}
\tag{251}
\mu\_n = \frac{\kappa\_0}{\kappa\_0 + n}\mu\_0 + \frac{n}{\kappa\_0 + n}\bar y
\end{equation}

\begin{equation}
\tag{252}
\kappa\_n = \kappa_0 + n
\end{equation}

\begin{equation}
\tag{253}
\nu\_n = \nu_0 + n
\end{equation}

\begin{equation}
\tag{254}
\Lambda\_n = \Lambda\_0 + S + \frac{\kappa\_0 n}{\kappa\_0 + n} (\bar x - \mu\_0)(\bar x - \mu\_0)^T,
\end{equation}

and

$S = \sum_{i=1}^n (x _i - \bar x) (x_i - \bar x)^T$

$\bar x = \frac{1}{n}\sum_{i=1}^n x_i$

$\bar y = \frac{1}{n}\sum_{i=1}^n y_i$


Here we clearly see the interpretation of $\kappa_0$ and $\nu_0$ as prior pseudo counts.
"""

# ╔═╡ 086f9a42-6a22-11eb-125f-f93352d7f5b9
md"""
In addition we have the diagonal variance σ and off-diagonal variance σₒ (we assume isotropic Gaussian).
"""

# ╔═╡ 2ed62778-6a1d-11eb-0717-419936c53ed6
md""" Variables
	κ₀ $(@bind κ₀ Slider(1:20))
	ν₀ $(@bind ν₀ Slider(2:100))
	σ $(@bind σ Slider(1:10))
	σₒ $(@bind σₒ Slider(0:0.1:1))
"""

# ╔═╡ 31cf0a66-6a0f-11eb-0f21-e57a762229f3
begin		
	μ₀ = zeros(2)
	Λ₀ = [
		σ σₒ;
		σₒ σ
	]
	prior = NormalInverseWishart(μ₀, κ₀, Λ₀, ν₀)
end;

# ╔═╡ 4927fa98-6a13-11eb-3d22-8f1336f1285d
begin
	plot_density(p, dist; args...) = begin
		x = y = range(-8, 8; length = 200)
		contour!(p, x, y, (x, y) -> pdf(dist, [x, y]); args...)
	end
	
	samples = vcat([rand(prior) for _ in 1:5])
	prior_samples_plt = plot(title = "Prior samples", xlabel = "x₁", ylabel = "x₂")
	for (i, s) in enumerate(samples)
		plot_density(prior_samples_plt, MvNormal(s...);
			color = i, cbar = nothing)
	end
	prior_samples_plt
end

# ╔═╡ 8a41ea0e-6a2f-11eb-1e3a-a1062d277644
md"""
Notice that the posterior predictive distribution is in fact not Gaussian, but is given by a multivatiare Student-T distribution
\begin{equation}
\tag{258}
p(x \vert \mathcal{D}) = t_{\nu - d + 1}\left (\mu, \frac{\Lambda(\kappa + 1)}{\kappa(\nu - d + 1)} \right )
\end{equation}
"""

# ╔═╡ f71d862e-6a3e-11eb-1806-5b9c0289e416
function posterior_predictive(x, post)
	d = size(x, 1)
	μ, Λchol, κ, ν = params(post)
	Λ = Λchol.L * Λchol.U
	df = ν - d + 1
	S = Λ*(κ + 1)/κ*df
	MvTDist(df, μ, S)
end;

# ╔═╡ fdea84fa-6a20-11eb-2430-572e73451bb2
begin
	x̄ = [2, 1.5] # data mean
	n = 50 # num samples
	x_t = rand(MvTDist(2, x̄, [1 0; 0 1]), n)
	post_t = posterior(prior, MvNormal, x_t)
	post_pred_t = posterior_predictive(x_t, post_t)
	post_pred_t_plt = plot(title = "Posterior pred. Student T data")
	plot_density(post_pred_t_plt, post_pred_t; cbar = nothing)
	scatter!(post_pred_t_plt, x_t[1,:], x_t[2,:], label = "Observations")
end

# ╔═╡ d20d5b14-6a42-11eb-0d3d-4b5717b732ff
md"""
Since the posterior predictive is a Student-T with fat tails it does not contract to capture Gaussian data as we might want. 
"""

# ╔═╡ 89b3e1e8-6a23-11eb-0b3f-85f46b749e4d
begin
	x_n = rand(MvNormal(x̄, [1 0; 0 1]), n)
	post_n = posterior(prior, MvNormal, x_n)
	post_pred_n = posterior_predictive(x_n, post_n)
	posterior_mean_pred_plt = plot(title = "Posterior pred. Gaussian data")
	plot_density(posterior_mean_pred_plt, post_pred_n; cbar = nothing)
	scatter!(posterior_mean_pred_plt, x_n[1,:], x_n[2,:], label = "Observations")
end

# ╔═╡ Cell order:
# ╠═5f4db9e4-6a26-11eb-2257-97e721b8727e
# ╠═1b3d1874-6a0f-11eb-1b72-a5e692d0c7e6
# ╠═31cf0a66-6a0f-11eb-0f21-e57a762229f3
# ╠═086f9a42-6a22-11eb-125f-f93352d7f5b9
# ╟─2ed62778-6a1d-11eb-0717-419936c53ed6
# ╠═4927fa98-6a13-11eb-3d22-8f1336f1285d
# ╠═8a41ea0e-6a2f-11eb-1e3a-a1062d277644
# ╠═f71d862e-6a3e-11eb-1806-5b9c0289e416
# ╠═fdea84fa-6a20-11eb-2430-572e73451bb2
# ╟─d20d5b14-6a42-11eb-0d3d-4b5717b732ff
# ╠═89b3e1e8-6a23-11eb-0b3f-85f46b749e4d
