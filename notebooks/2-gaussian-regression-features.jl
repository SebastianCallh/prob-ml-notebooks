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
	using Zygote: gradient
	Random.seed!(1231245)
	σ = .5
	N = 50
	X = collect(range(-6, 6; length = N)) .+ randn(N)
	y = X .+ randn(N)
	y[10:30] .= y[10:30] .+ 4
	y[30:40] .= sin.(y[30:40]) 
	scatter(X, y, title = "Observed data", 
		xlabel = "X", ylabel = "y",
		label = nothing
	)
end

# ╔═╡ 67c6886a-6a4b-11eb-377b-278d91b6d03f


# ╔═╡ b6cfc0c6-6dfb-11eb-058d-db10ed33d71a
md"""
## Bayesian regression with learned features
In the last notebook we saw how to compute the posterior under a Gaussian prior and likelihood when fitting a linear model. The model we fit was doubly linear: it was linear in model weights $w$, and we actually fit a line. However, the posterior did not capture the data and was poorly calibrated.
We will leave the calibration issue for a bit and focus on improving the model fit. To thie end we are going to see just how flexible the model can be for various $\phi$, and how to learn feature representations. 

We are going to use the same setting as in the previous notebook. Recall that $X \in \mathbb{R}^{D \times N}$ is the data matrix with $N$ observations of dimension $D$ and $\mathbf{y} \in \mathbb{R}^N$ the observed values.

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

	"""Here we use the alterantive posterior identity since K < N."""
	function posterior_f(reg::GaussianRegression, x)
		@unpack X, y, μ, Σ, σ, ϕ = reg
		ϕX = ϕ(X)
		ϕx = ϕ(x)
		invΣ = inv(Σ)
		A = Symmetric(invΣ + σ^(-2)*ϕX*ϕX')
		G = cholesky(A)
		A = (G\ϕx)'
		μₙ = A*(invΣ*μ + σ^(-2)*ϕX*y)
		Σₙ = A*ϕx
		μₙ, Σₙ
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
	sigmoid(xx; θ₂ = 1) = 
			x -> feature_grid(x′ -> 1 ./ (1 .+ exp.(.-(x .- x′)./θ₂)), xx)
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

Feature function $(@bind feature_name Select(collect(keys(ϕs))))
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

Number of features $(@bind num_features_input2 Slider(2:25; show_value=true))

Feature function $(@bind feature_name2 Select(collect(keys(ϕs))))

"""

# ╔═╡ 9a7dd170-6eb3-11eb-11a2-23759c9808fe
begin
	feature_xx = range(xlim..., length = num_features_input2)
	reg₂ = GaussianRegression(
		X, y, zeros(num_features_input2),
		diagm(ones(num_features_input2)), σ,
		features(feature_name2, feature_xx)
	)
	plot_features(reg₂, num_features_input2)
end

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
So how do we find $\hat \theta$? A simple solution to this is to pick $\theta$ that maximises the marginal likelihood (evidence) $p(y \vert X, \theta)$. As you probably realise, this is a point estimate, but at least it is tractable. It is very similar to maximum likelihood, but instead of maximising $p(y \vert X, f, \theta)$ directly, we maximise $p(y \vert X, \theta)$. When done this way it is referred to as *type-II maximum likelihood*. To do this we optimise the expression 

```math
\begin{equation}
\begin{split}
\hat \theta
= & \text{arg\,max}_\theta\; p(y \vert X, \theta) \\
= & \text{arg\,max}_\theta\; \mathcal{N}(y; \phi_X^{\theta^T}\mu, \phi_X^{\theta^T} \Sigma \phi_X^\theta+ \sigma^2 I) \\
= & \text{arg\,min}_\theta\; - \log \mathcal{N}(y; \phi_X^{\theta^T}\mu, \phi_X^{\theta^T} \Sigma \phi_X^\theta+ \sigma^2 I) \\
= & \text{arg\,min}_\theta\; \frac{1}{2} 
\left( \left( \mathbf{y} - \phi_X^\theta\mu \right)^T
\left( \phi_X^{\theta^T} \Sigma \phi_X^\theta + \Lambda \right)^{-1}
\left( \mathbf{y} - \phi_X^\theta\mu \right) 
+ \log \vert \phi_X^{\theta^T} \Sigma \phi_X^\theta + \Lambda \vert \right )
+ \frac{N}{2} \log 2\pi \\
\end{split}
\end{equation}
```

In other words, we minimise the negative log marginal likelihood. Let us do this to find good locations for the location of the feature functions $\theta = x \prime$. 
A convenient way of doing this is through [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), which requires us to take gradients of $p(y \vert X, \theta)$ with respect to $\theta$. Luckily, we live in the era of automatic differentiation, and Julia has *great* libraries to this end. We are going to utilize the `gradient` function from the [Zygote](https://fluxml.ai/Zygote.jl/) package to compute the gradient of our loss. On caveat with this is that we cannot differentiate through discontiuities, so the step function features have to go for now. You can use the by now familiar sliders below to fit different features. Note that this optimisation is not convex so you might get stuck in local minima.

Number of features $(@bind input_k3 Slider(2:10; show_value=true))

Feature function $(@bind input_feature3 Select(collect(filter(k-> k != "step functions", keys(ϕs)))))
"""

# ╔═╡ f0ef960a-6ecd-11eb-2a55-7d447bea3af4
begin
	"""This is actually not exactly the marginal likelihood, but only proportional to it since we dropped the constants. We do not need those for optimisation."""
	function marginal_likelihood(X, y, μ, Σ, Λ)
		θ -> begin
			K = length(θ)
			ϕ = features(input_feature3, θ)
			ϕX = ϕ(X)
			κXX = ϕX'*Σ*ϕX
			r = y .- ϕX'*μ
			r'*inv(κXX + Λ)*r + log(det(κXX + Λ))
		end
	end
	
	function fit(loss, θ; steps, α = 0.001)
		losses = zeros(steps)
		for i in 1:steps
			losses[i] = loss(θ)
			g = gradient(loss, θ)[1]
			θ -= α*g
		end
		θ, losses
	end
	K = input_k3
	θ₀ = collect(range(-10, 10, length = K)) .+ randn(K)
	loss = marginal_likelihood(X, y, ones(K), diagm(ones(K)), σ^2*I(length(X)))
	θ̂, losses = fit(loss, θ₀; steps = 100)
end;

# ╔═╡ f7ae77f2-6ed9-11eb-33f7-db8775acec19
begin
	reg₃ = GaussianRegression(
		X, y, zeros(K), diagm(ones(K)), σ,
		features(input_feature3, θ̂)
	)
	
	xx = collect(range(xlim..., length = 200))
	μₙ, Σₙ = posterior_f(reg₃, xx)
	posterior_plt  = plot(xx, μₙ, ribbon = 2*sqrt.(diag(Σₙ)), 
		color = 2, label = "p(f | X, y)", legend =:topleft)
	scatter!(posterior_plt, X, y, label = "Observations",
		color = 1,
		title ="Posterior predictive",
		xlabel = "x", ylabel = "f(x)")
	scatter!(posterior_plt, θ̂, -5 .* ones(length(θ̂)),
		markershape = :uptriangle,
		label = "Feature locations")
	
	loss_plt = plot(losses, title = "Loss curve",
		label = nothing, xlabel = "Iteration", ylabel = "Loss")
	plot(loss_plt, posterior_plt, layout = (2, 1), size = (600, 600))
end

# ╔═╡ 0db1d1bc-6e3c-11eb-0b32-b9a56eb0b2e6
md"""
By maximising the marginal likelihood we are able to learn good features for our data which we can then use to perform Gaussian inference. The learned features are noticably better than the uniformly placed ones, and significantly fewer than assigning one feature per data point. Despite this the posterior captures the data very well.

### Ending notes.
In this notebook we expanded the notion of Gaussian inference to more flexible function classes by using various feature functions. We also saw an example of feature learning, where we learn the position of the features using type-II maximum likelihood. Even though we only learned the feature positions, we can use the exact same procedure for any hyper-parameter, as long as we can take gradients which is made super easy thanks to automatic differentiation. Indeed, this is the idea that leads us to deep learning.
"""

# ╔═╡ Cell order:
# ╟─67c6886a-6a4b-11eb-377b-278d91b6d03f
# ╠═b6cfc0c6-6dfb-11eb-058d-db10ed33d71a
# ╠═23451f6e-6e1d-11eb-35fb-1b560a12e694
# ╠═79934a56-6e1c-11eb-0728-1b6a25618285
# ╠═e34a66c8-6e9c-11eb-1ab3-392857ed152b
# ╟─0f529eae-6ea5-11eb-280c-9106b55e81fe
# ╟─659f63b2-6ea7-11eb-2333-35d3f51a748c
# ╟─5114873e-6dfc-11eb-0da3-21a6186d9da2
# ╟─4acd1fa0-6eb3-11eb-15b2-bf84b53f3880
# ╠═9a7dd170-6eb3-11eb-11a2-23759c9808fe
# ╟─f93023b0-6eb5-11eb-189a-a7951ec0431e
# ╟─574db804-6ec0-11eb-205c-65ad68022b58
# ╟─f0ef960a-6ecd-11eb-2a55-7d447bea3af4
# ╟─f7ae77f2-6ed9-11eb-33f7-db8775acec19
# ╠═0db1d1bc-6e3c-11eb-0b32-b9a56eb0b2e6
