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

# ╔═╡ 23451f6e-6e1d-11eb-35fb-1b560a12e694
begin
	using Random, Distributions, LinearAlgebra, Parameters, Plots, PlutoUI
	Random.seed!(1231245)
	σ = 1.5
	N = 50
	X = collect(range(-6, 6; length = N)) .+ randn(N)
	y = X .+ randn(N)
	y[10:30] .= y[10:30] .+ 4
	y[30:40] .= sin.(y[30:40]) 
	scatter(X, y, title = "Observed data", xlabel = "X", ylabel = "y", label = nothing)
end

# ╔═╡ 67c6886a-6a4b-11eb-377b-278d91b6d03f


# ╔═╡ b6cfc0c6-6dfb-11eb-058d-db10ed33d71a
md"""
## Bayesian regression with various features
In the last notebook we saw how to compute the posterior under a Gaussian prior and likelihood when fitting a linear model (linear in model weights $w$). However, the data fit was very bad and the model was poorly calibrated. We will leave the calibration issue for a bit and focus on improving the model fit by considering various feature functions $\phi$. Recall that $X \in \mathbb{R}^{D \times N}$ is the data matrix with $N$ observations of dimension $D$ and $\mathbf{y} \in \mathbb{R}^N$ the observed values.

[Probabilistic ML - Lecture 8 - Learning Representations](https://www.youtube.com/watch?v=Zb0K_S5JJU4&t=1142s)
"""

# ╔═╡ 79934a56-6e1c-11eb-0728-1b6a25618285
md"""
Before we move on, here is cleaned-up version of the code from the last notebook.
"""

# ╔═╡ e34a66c8-6e9c-11eb-1ab3-392857ed152b
begin
	@with_kw struct GaussianRegression{T<:Real, F}
		X::Vector{T}
		y::Vector{T}
		μ::Vector{T}
		Σ::Matrix{T}
		σ::T
		ϕ::F
	end
	
	function prior_f(reg::GaussianRegression, x)
		@unpack μ, Σ, ϕ = reg
		ϕₓ = ϕ(x)
		m = ϕₓ'*μ
		s = ϕₓ'*Σ*ϕₓ
		m, s
	end
	
	function prior_sample_f(reg::GaussianRegression, x, kwargs...)
		@unpack μ, Σ, ϕ = reg
		pw = MvNormal(μ, Σ)
		w = rand(pw, kwargs...)
		f = ϕ(x)'*w
	end
	
	function posterior_f(reg::GaussianRegression, x)
		@unpack X, y, μ, Σ, σ, ϕ = reg
		ϕX = ϕ(X)
		ϕx = ϕ(x)
		κxX = ϕx'*Σ*ϕX
		κXX = ϕX'*Σ*ϕX
		κxx = ϕx'*Σ*ϕx
		r = y - ϕX'*μ
		A = κXX + σ^2*I(N)
		G = cholesky(A)
		A = (G\κxX')'
		m = ϕx'*μ + A*r
		s = κxx - A*κxX'
		m, s
	end
	
	function evidence(reg::GaussianRegression, x, y)
		@unpack μ, Σ, σ, ϕ = reg
		ϕX = ϕ(X)
		κXX = ϕX'*Σ*ϕX
		py = MvNormal(ϕX'*μ, Symmetric(κXX + σ^2*I(length(x))))
		pdf(py, y)
	end
end

# ╔═╡ 659f63b2-6ea7-11eb-2333-35d3f51a748c
begin
	xlim = -10, 10
	feature_grid(f, xx) = mapreduce(f, hcat, xx)'
	step(xx) = x -> feature_grid(x′ -> Int.(x .>= x′), xx)
	gaussian(xx; θ₁ = 10, θ₂ = 4) = 
			x -> feature_grid(x′ -> θ₁.*exp.(-(x .- x′).^2)./2θ₂, xx)
	linear(xx) = x -> feature_grid(x′ -> abs.(x .- x′) .- x′, xx)
	sigmoid(xx; θ₁ = 1, θ₂ = 2) = 
			x -> feature_grid(x′ -> 1 ./ (1 .+ exp.(.-(x .- x′ .- θ₁)./θ₂)), xx)
	relu(xx) = x -> feature_grid(x′ -> max.(x, x′), xx)

	ϕs = Dict(
		"step functions" => step,
		"Gaussian" => gaussian,
		"piece-wise linear" => linear,
		"sigmoid" => sigmoid,
		"ReLU" => relu
	)
	features(name, grid) = ϕs[name](grid)
end

# ╔═╡ 0f529eae-6ea5-11eb-280c-9106b55e81fe
md"""
Remember that the only assumptions we made for the whole inference framework to work was a Gaussian prior and likelihood, and that the function values are a linear (or affine) map of the model weights. No assumptions were made regarding $\phi$ and so we are free to choose this function as we please, and inference will still work as before. In the list below you can choose various feature functions and see how they affect the posterior.

Feature function ϕ $(@bind feature_name Select(collect(keys(ϕs))))
"""

# ╔═╡ 5114873e-6dfc-11eb-0da3-21a6186d9da2
begin
	function plot_features(reg, K)
		xx = collect(range(xlim..., length = 200))
		μ₀, Σ₀ = prior_f(reg, xx)
		fs = prior_sample_f(reg, xx, 5)
		prior_plt = plot(xx, μ₀, ribbon=2*sqrt.(diag(Σ₀)), 
			title = "Prior predictive",
			xlabel = "x",
			ylabel = "p(f)",
			legend = :topleft,
			label = nothing
		)
		for (i, f) in enumerate(eachcol(fs))
			plot!(prior_plt, xx, f, color = 3,
				label = i == 1 ? "Prior sample" : nothing)
		end

		scatter!(prior_plt, X, y, label = "Observations", color = 1)

		μₙ, Σₙ = posterior_f(reg, xx)
		posterior_plt  = plot(xx, μₙ, ribbon = 2*sqrt.(diag(Σₙ)), 
			color = 2,
			label = "p(f | X, y)", legend =:topleft)
		scatter!(posterior_plt, X, y, label = "Observations",
			color = 1,
			title ="Posterior predictive",
			xlabel = "x", ylabel = "f(x)")
	
		plot(prior_plt, posterior_plt, layout = (2, 1), size = (600, 600))
	end
	
	reg₁ = GaussianRegression(
		X, y, zeros(N), diagm(ones(N)), .5,
		features(feature_name, X)
	)
	plot_features(reg₁, N)
end

# ╔═╡ 4acd1fa0-6eb3-11eb-15b2-bf84b53f3880
md"""
Hopefully you see how flexible the  Gaussian inference framework is. Since we are not tied to any function class we can simply plug in one that is better suited for any particular problem. While we have been limiting ourselves to using features of our input points $\phi(X)$ we are in fact free to use features $\phi(x\prime)$ of any input point $x\prime$ of our choice. You can play around with the slider below to create a grid of features which are used to fit the model to see the impact of this.

Number of features K $(@bind input_K Slider(2:25))

Feature function ϕ $(@bind feature_name2 Select(collect(keys(ϕs))))

"""

# ╔═╡ 9a7dd170-6eb3-11eb-11a2-23759c9808fe
begin
	feature_xx = range(xlim..., length = input_K)
	reg₂ = GaussianRegression(
		X, y, zeros(input_K), diagm(ones(input_K)), σ,
		features(feature_name2, feature_xx)
	)
	plot_features(reg₂, input_K)
end

# ╔═╡ 82017c8c-6eb8-11eb-03b2-0333cc33ac48
md"""
Model evidence $(evidence(reg₂, X, y))
"""

# ╔═╡ f93023b0-6eb5-11eb-189a-a7951ec0431e
md"""
As you may have noticed, using more features allows for a richer model which better captures the data. And while more flexibility is great, we have now introduced several new steps to our modeling: We have to select $\phi$ and $x\prime$. 
Note that nothing forces us to use a grid for $x \prime$; we just do it here for convenience. If we were serious about leraning this function we might instead want to place more features in regions where the function changes rapidly, and fewer where not much is going on.

An additional complication which we so far swept under the rug is that $\phi$ can have parameters $\theta$ of its own. In fact, the Gaussian and sigmoid function in the list above have two parameters each. How do we deal with all these new quantities? Let us take a step back and think about how this fits in our inference framework.

### Hierarchical Bayesian inference
So far the object of interest for us has been
```math
p(f \vert X, y) = \frac{p(y \vert f, X) p(f)}{\int p(y \vert f)p(f) df}
```

but given our new unknown quantities $\theta$ we would like to include them into our prior and compute an extended posterior
```math
p(f \vert X, y, \theta) = \frac{p(y \vert f, X, \theta) p(f \vert \theta)}{\int p(y \vert f, X,\theta) p(f \vert \theta) df}
= \frac{p(y \vert f, X, \theta) p(f \vert \theta)}{p(y \vert X, \theta)}.
```
We refer to this as *hierarchical* inference, since we now have two levels of parameters. It turns out that the evidence for the model over $\theta$ is 
```math
p(y \vert X, \theta) = \mathcal{N}(y; \phi_X^{\theta^T}\mu, \phi_X^{\theta^T} \Sigma \phi_X^\theta + \Lambda),
```
where $\phi_X^\theta =\phi(X;\theta)$ and $\Lambda$ is the observation noise.

Unfortunately this does not have a closed form, since $\theta$ enters in a non-linear fashion through $\phi$. This is common when doing Bayesian inference; only the simplest of cases have closed form posteriors. So what can we do to proceed? We saw previously that we were able to succesfully compute the posterior for a given $\theta$, and by fiddling around with different values we could find better or worse values. While it is not properly Bayesian, we could try to find the single "best" parametrisation $\hat \theta$ and use it for inference. In this setting we treat $\theta$ as a *hyper-parameter*, which differs from our *parameters* $w$ which in that we do not include them in our posterior. They are in a sense at the "top level" of our model. 
"""

# ╔═╡ 574db804-6ec0-11eb-205c-65ad68022b58
md"""
#### Type-II maximum likelihood
So how do we find $\hat \theta$? A simple solution to this is to pick $\theta$ that maximises the marginal likelihood (evidence) $p(y \vert X, \theta)$. As you probably realise, this is a point estimate, but at least it is tractable. It is very similar to maximum likelihood, the difference being that we are not maximising $p(y \vert X, f, \theta)$ directly, but instead $p(y \vert X, \theta)$. When done this way it is referred to as *type-II maximum likelihood*. The full expression we want to optimise is

```math
\begin{equation}
\begin{split}
\hat \theta
= & \text{arg\,max}_\theta\; p(y \vert X, \theta) \\
= & \text{arg\,max}_\theta\; \mathcal{N}(y; \phi_X^{\theta^T}\mu, \phi_X^{\theta^T} \Sigma \phi_X^\theta+ \sigma^2 I) \\
= & \text{arg\,min}_\theta\; - \log \mathcal{N}(y; \phi_X^{\theta^T}\mu, \phi_X^{\theta^T} \Sigma \phi_X^\theta+ \sigma^2 I) \\
= & \text{arg\,min}_\theta\; \frac{1}{2} 
\left( \left( \mathbf{y} - \phi_X^\theta\mu \right)^T
\left( \phi_X^{\theta^T} \Sigma \phi_X^\theta \right)^{-1}
\left( \mathbf{y} - \phi_X^\theta\mu \right) 
+ \log \vert \phi_X^{\theta^T} \Sigma \phi_X^\theta + \Lambda \vert \right )
+ \frac{N}{2} \log 2\pi.
\end{split}
\end{equation}
```
In other words, we want to minimise the negative log marginal likelihood.
A convenient way of doing this is through gradient descent, which requires us to take gradients of $p(y \vert X, \theta)$ with respect to $\theta$.
"""

# ╔═╡ f0ef960a-6ecd-11eb-2a55-7d447bea3af4
begin
	
end

# ╔═╡ 0db1d1bc-6e3c-11eb-0b32-b9a56eb0b2e6
md"""
### Ending notes
We have seen how to infer model weights $w$ in closed form using Gaussian distributions. While we used a linear regression, the only requirement we had was that the model was linear *in the weights*.

We also saw how the model is surprisingly overconfident. This can be explained by the fact that the posterior uncertainty does not depend on observations $y$, but only on $x$. There is simply no information that says that the prediction is far away from the observed values that can be taken into account. Let us think about how to improve the model fit in the next notebook.
"""

# ╔═╡ 66176d80-6ea7-11eb-13a8-f5d4927aa0d2


# ╔═╡ Cell order:
# ╟─67c6886a-6a4b-11eb-377b-278d91b6d03f
# ╠═b6cfc0c6-6dfb-11eb-058d-db10ed33d71a
# ╠═23451f6e-6e1d-11eb-35fb-1b560a12e694
# ╠═79934a56-6e1c-11eb-0728-1b6a25618285
# ╠═e34a66c8-6e9c-11eb-1ab3-392857ed152b
# ╠═0f529eae-6ea5-11eb-280c-9106b55e81fe
# ╠═659f63b2-6ea7-11eb-2333-35d3f51a748c
# ╠═5114873e-6dfc-11eb-0da3-21a6186d9da2
# ╠═4acd1fa0-6eb3-11eb-15b2-bf84b53f3880
# ╠═9a7dd170-6eb3-11eb-11a2-23759c9808fe
# ╠═82017c8c-6eb8-11eb-03b2-0333cc33ac48
# ╠═f93023b0-6eb5-11eb-189a-a7951ec0431e
# ╠═574db804-6ec0-11eb-205c-65ad68022b58
# ╠═f0ef960a-6ecd-11eb-2a55-7d447bea3af4
# ╠═0db1d1bc-6e3c-11eb-0b32-b9a56eb0b2e6
# ╠═66176d80-6ea7-11eb-13a8-f5d4927aa0d2
