### A Pluto.jl notebook ###
# v0.12.21

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

# ╔═╡ d8f7dde6-744f-11eb-027a-43075e3bf744
using PlutoUI

# ╔═╡ b6cfc0c6-6dfb-11eb-058d-db10ed33d71a
md"""
## Bayesian regression with learned features
In the last notebook we saw how to compute the posterior under a Gaussian prior and likelihood when fitting a linear model. The model we fit was doubly linear: it was linear in model weights $w$, and we fit an actual line to the data. However, the posterior did not capture the data and was poorly calibrated (in our case, overly confident).
We will leave the calibration issue for a bit and focus on improving the model fit. To this end we are going to see just how flexible the model can be for various $\phi$, and how to learn feature representations. 

We are going to use the same setting as in the previous notebook. Recall that $X \in \mathbb{R}^{D \times N}$ is the data matrix with $N$ observations of dimension $D$ and $\mathbf{y} \in \mathbb{R}^N$ the observed values.

Relevant lecture: [Probabilistic ML - Lecture 8 - Learning Representations](https://www.youtube.com/watch?v=Zb0K_S5JJU4&t=1142s)
"""

# ╔═╡ 659f63b2-6ea7-11eb-2333-35d3f51a748c
begin
	step(θ) = x -> Int.(x' .>= θ)'
	gaussian(θ) = x -> exp.(-(x' .- θ).^2)'./2	
	linear(θ) = x -> (abs.(x' .- θ) .- θ)'
	sigmoid(θ) = x -> 1 ./ (1 .+ exp.(.-(x' .- θ)))'
	relu(θ) = x -> max.(0., x' .- θ)'

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

# ╔═╡ 4acd1fa0-6eb3-11eb-15b2-bf84b53f3880
md"""
Hopefully you see how flexible the  Gaussian inference framework is. Since we are not tied to any function class we can simply plug in one that is better suited for any particular problem. While we have been limiting ourselves to using features of our input points $\phi(X)$ we are in fact free to use features $\phi(x\prime)$ of any input point $x\prime$ of our choice (you can think of this as as bias term). You can play around with the slider below to create a grid of features which are used to fit the model to see the impact of this. Since everything is Gaussian, we can also compute the model evidence as a way of evaluating the model fit. The higher the evidence, the more probable the data is to be observed under the model. Hence, we can use the evidence as a way to perform model selection.

Number of features $(@bind num_features_input2 Slider(2:25; show_value=true))

Feature function $(@bind feature_name2 Select(collect(keys(ϕs))))

"""

# ╔═╡ f93023b0-6eb5-11eb-189a-a7951ec0431e
md"""
As you may have noticed, using more features allows for a richer model which better captures the data. And while more flexibility is great, we have now introduced several new steps to our modeling: We have to select $\phi$ and $x\prime$. 
Note that nothing forces us to use a grid for $x \prime$; we just do it here for convenience. If we were serious about leraning this function we might instead want to place more features in regions where the function changes rapidly, and fewer where not much is going on. An additional complication which we so far swept under the rug is that $\phi$ can have parameters $\theta$ of its own. How do we deal with all these new quantities? Let us take a step back and think about how this fits in our inference framework.

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
A convenient way of doing this is through [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), which requires us to take gradients of $p(y \vert X, \theta)$ with respect to $\theta$. Luckily, we live in the era of automatic differentiation, and Julia has *great* libraries to this end. We are going to utilize the `gradient` function from the [Zygote](https://fluxml.ai/Zygote.jl/) package to compute the gradient of our loss. On caveat with this is that we cannot differentiate through discontiuities, so the step function features have to go for now. You can use the by now familiar sliders below to fit different features. Note that this optimisation is not convex so you will probably find different minimas each run.

Number of features $(@bind input_k3 Slider(2:20; show_value=true))

Feature function $(@bind input_feature3 Select(collect(filter(k-> k != "step functions", keys(ϕs)))))
"""

# ╔═╡ 0db1d1bc-6e3c-11eb-0b32-b9a56eb0b2e6
md"""
By maximising the marginal likelihood we are able to learn good features for our data which we can then use to perform Gaussian inference. Even though the learned features are significantly fewer than the number of data point, their positions lets them capture the data well.

### Ending notes.
In this notebook we expanded the notion of Gaussian inference to more flexible function classes by using various feature functions. We also saw an example of feature learning, where we learn the position of the features using type-II maximum likelihood. Even though we only learned the feature positions in our example, we can use the exact same procedure for any hyper-parameter, as long as we can computer their gradients. Thanks to automatic differentiation, this can be done automatically for us. Feature learning using gradients and automatic differentiation is a very powerful framework, and the key idea behind deep learning. In the next notebook we are going to take a probabilistic perspective on deep learning and see how it relates to what we have seen so far.  
"""

# ╔═╡ dca94b3a-7122-11eb-3080-fdbc8581823a
begin
	"""This is just to include external files. Please ignore it"""
	function ingredients(path::String)
		# this is from the Julia source code (evalfile in base/loading.jl)
		# but with the modification that it returns the module instead of the last object
		name = Symbol(basename(path))
		m = Module(name)
		Core.eval(m,
	        Expr(:toplevel,
	             :(eval(x) = $(Expr(:core, :eval))($name, x)),
	             :(include(x) = $(Expr(:top, :include))($name, x)),
	             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
	             :(include($path))))
		m
	end
	
	GR = ingredients("../src/gaussian_regression.jl")
	Data = ingredients("../src/data.jl")
end

# ╔═╡ 23451f6e-6e1d-11eb-35fb-1b560a12e694
begin
	using Random, Distributions, LinearAlgebra, Parameters, Plots
	using Zygote: gradient

	X, y = Data.linreg_toy_data()
	D, N = size(X)
	xlim = -10, 10
	ylim = -12, 12
	scatter(
		X', y, title = "Observed data", 
		xlabel = "X", ylabel = "y",
		label = nothing
	)
end

# ╔═╡ 5114873e-6dfc-11eb-0da3-21a6186d9da2
begin
	xx = collect(range(xlim..., length = 200))'
	reg₁ = GR.GaussianRegression(
		X, y, zeros(N), diagm(ones(N)), .5,
		features(feature_name, X)
	)
	GR.plot_features(reg₁, xx, X, xlim, ylim)
end

# ╔═╡ 7d4fae44-74ed-11eb-254e-f7bde69e6eee
md"""
Model log evidence: $(GR.log_evidence(reg₁))
"""

# ╔═╡ f0ef960a-6ecd-11eb-2a55-7d447bea3af4
begin
	function fit(loss, θ; steps, α = 0.001)
		losses = zeros(steps)
		for i in 1:steps
			losses[i] = loss(θ)
			∇θ = first(gradient(loss, θ))
			θ -= α*∇θ
		end
		θ, losses
	end
	
	K = input_k3
	σ = 0.5
	μ = zeros(K)
	Σ = diagm(ones(K))
	ϕ = ϕs[input_feature3]
	θ₀ = collect(range(-10, 10, length = K))' .+ randn(K)'
	θ̂, losses = fit(
		θ -> -GR.log_evidence(GR.GaussianRegression(X, y, μ, Σ, σ, ϕ(θ))), 
		θ₀; steps = 100
	)
end;

# ╔═╡ 9a7dd170-6eb3-11eb-11a2-23759c9808fe
begin
	feature_xx = range(xlim...; length = num_features_input2)'
	reg₂ = GR.GaussianRegression(
		X, y, zeros(num_features_input2),
		diagm(ones(num_features_input2)), σ,
		features(feature_name2, feature_xx)
	)
	plt = GR.plot_features(reg₂, xx, feature_xx, xlim, ylim)
end

# ╔═╡ 4f3a8702-74ed-11eb-1732-bf79fe482cec
md"""
Model log evidence: $(GR.log_evidence(reg₂))
"""

# ╔═╡ f7ae77f2-6ed9-11eb-33f7-db8775acec19
begin
	reg₃ = GR.GaussianRegression(
		X, y, μ, Σ, σ,
		features(input_feature3, θ̂)
	)
	pf = GR.posterior_f(reg₃, xx)
	posterior_plt = plot(xx', pf.μ, ribbon = 2*sqrt.(diag(pf.Σ)), 
		color = 2, label = "p(f | X, y)", legend =:topleft,
		xlim = xlim, ylim=(-10, 10))
	scatter!(posterior_plt, X', y, label = "Observations",
		color = 1,
		title ="Posterior predictive",
		xlabel = "x", ylabel = "f(x)")
	scatter!(posterior_plt, θ̂', -9.6 .* ones(length(θ̂)),
		markershape = :uptriangle,
		label = "Feature locations")

	loss_plt = plot(losses, title = "Loss curve",
		label = nothing, xlabel = "Iteration", ylabel = "Log marginal likelihood")
	plot(loss_plt, posterior_plt, layout = (2, 1), size = (600, 600))
end

# ╔═╡ Cell order:
# ╠═d8f7dde6-744f-11eb-027a-43075e3bf744
# ╠═b6cfc0c6-6dfb-11eb-058d-db10ed33d71a
# ╠═23451f6e-6e1d-11eb-35fb-1b560a12e694
# ╟─0f529eae-6ea5-11eb-280c-9106b55e81fe
# ╠═659f63b2-6ea7-11eb-2333-35d3f51a748c
# ╟─7d4fae44-74ed-11eb-254e-f7bde69e6eee
# ╠═5114873e-6dfc-11eb-0da3-21a6186d9da2
# ╟─4acd1fa0-6eb3-11eb-15b2-bf84b53f3880
# ╟─4f3a8702-74ed-11eb-1732-bf79fe482cec
# ╠═9a7dd170-6eb3-11eb-11a2-23759c9808fe
# ╟─f93023b0-6eb5-11eb-189a-a7951ec0431e
# ╟─574db804-6ec0-11eb-205c-65ad68022b58
# ╠═f0ef960a-6ecd-11eb-2a55-7d447bea3af4
# ╠═f7ae77f2-6ed9-11eb-33f7-db8775acec19
# ╠═0db1d1bc-6e3c-11eb-0b32-b9a56eb0b2e6
# ╟─dca94b3a-7122-11eb-3080-fdbc8581823a
