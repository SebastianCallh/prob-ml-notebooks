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

# ╔═╡ a2ac2eac-7111-11eb-2101-b198c13d3f5f
md"""
# Deep learning from a probabilistic perspective
We saw in the previous notebook that we can use type-II maximum likelihood to learn parmaeters $\thete$ of our feature function $\phi$. Idealy we would like to marginalise over them, but due to computational constraints we resort to point-estimates. That's all well and good, but perhaps it is not expressive enough for very complex data. In such a situation we might want to look at deep learning.

### Neural networks are feature function learners
There are a lot of moving pieces when training modern neural networks, and a lot of empirical work has been done on how to properly mix stochastic optimizers, batch normalisation, self-attention, weight decay etc. to train good neural networks. In this notebook we are going to see that below all the complexities lies a very simple structure, which approximates (albeit crudely) the Bayesian posterior over the neural network weights.

Deep learning is at it's core very similar to what we have done previously, and also assumes that the target function can be represented by a linear combination of a set of feature functions $y = \phi(x; \theta)^Tw$. However, the way features are computed is more sophisticated. Instead of learning a single feature function $\phi(x; \theta)^Tw$ we learn a composition of $\ell$ features $\phi_\ell = \phi_\ell(x; \theta_\ell)^T$. To see this let us fit a small neural network on our toy data.
We are going to use the same setting as in the previous notebook. Recall that $X \in \mathbb{R}^{D \times N}$ is the data matrix with $N$ observations of dimension $D$ and $\mathbf{y} \in \mathbb{R}^N$ the observed values.

[Probabilistic ML - Lecture 8 - Learning Representations](https://www.youtube.com/watch?v=Zb0K_S5JJU4&t=1142s)
"""

# ╔═╡ f88bc10a-75cf-11eb-001b-35d72fde997b
md"""
As we might expect, the prior induced by the neural network is extremely flexible (observe the scale on the y-axis) and it fits the data without issues. In this example we only learn the locations of the featues, but we could imagine learning both multiplicative weights and specific parameters for the individual $\phi_\ell$ in the same fashion.
"""

# ╔═╡ 4b90cb7e-6f00-11eb-2fe7-af7d99ad7ed9
md"""
### Weight decay
Now you might say "But that's just a toy example, what about *proper* deep learning? Fair point. We are only learning the location (the bias term) of our features on toy data. However, it turns out that there is general Bayesian interpretation
to the type of inference commonly used when training "real" neural networks as well. Remember the goal of Bayesian inference: computing the posterior

```math
p(\theta \vert x, y) = \frac{p(y \vert x, \theta) p(\theta)}{\int p(y \vert \theta)p(\theta)}.
```

Since the integral in the denominator typically does not have a closed form, our main concern is how to create good posterior approximations.
One very crude approximation is to disregard the uncertainty in the weights and simply compute the *maximum a posteriori* (MAP) estimate

```math
\theta_{MAP} = \text{arg max}_\theta \;p(\theta \vert x, y)
```

which lets us ignore the denominator since it is only a normalization constant, and instead of performing difficult integration we can instead optimise

```math
\begin{equation}
\begin{split}
\text{arg max}_\theta \; p(y \vert x, \theta) p(\theta) 
= \text{arg min}_\theta \; - \log p(y \vert x, \theta) - \log p(\theta),
\end{split}
\end{equation}
```
where we switch signs of the the optimization and take the logarithm, which is a monotonic function and hence does not shift the minima.
If we assume $p(y \vert x, \theta) = \mathcal{N}(y; f(x), \sigma^2I)$ and prior $p(\theta) = \mathcal{N}(\theta; 0, \sigma_0^2I)$ we get

```math
\begin{equation}
\begin{split}
- \log p(y \vert x, \theta) - \log p(\theta) 
& =  \log \mathcal{N}(y; f(x) \vert \theta) & - \log \mathcal{N}(\theta) \\
& =  \frac{1}{2\sigma^2}\sum_{i=1}^N(y_i - f(x_i))^2 & + \frac{1}{2\sigma_0^2}\sum_{j=1}^D \theta_j^2 + \text{const}.
\end{split}
\end{equation}
```

So far so good. If we now multiply this quantity by $2\sigma^2$ (this does not change minima) we get

```math
- \log p(\theta \vert \mathcal{D}) \propto \sum_{i=1}^N(y_i - \hat y_i)^2 + \frac{\sigma^2}{\sigma_0^2} \sum_{j=1}^D \theta_j^2
```

where we can identify $\frac{\alpha}{2} = \frac{\sigma^2}{\sigma_0^2}$ to recover the classic weight decay term $\frac{\sigma^2}{\sigma_0^2} \sum_{j=1}^D\theta_j^2 = \frac{\alpha}{2} \theta^T\theta$ (equation 7.15 in the [deep learning book](https://www.deeplearningbook.org/contents/regularization.html)).
In other words, the weight decay is the ratio between the observation noise variance and the prior variance.

With this insight we can make some interesting observations. If the weight decay $\alpha$ is large $\sigma_0^2$ is small in relation to $\sigma^2$, which reflects a prior belief that $\theta$ is small, which is sensible given that we want to regularise. We can also see that $lim_{\sigma_0 \to \infty} \frac{\sigma^2}{\sigma_0^2} = 0$
recovers a uniform prior over $\theta$ and the maximum likelihood estimate. You can visualise this in the plot below.

σ $(@bind input_σ Slider(0.1:0.1:3; default=0.5, show_value=true))

σ₀ $(@bind input_σ₀ Slider(0.5:0.1:3; default=0.5, show_value=true)) 

"""

# ╔═╡ 278e0f08-6fbd-11eb-25e8-5b73d8d7ed6d
md"""
### What about classification?
The above case only covered regression, but neural networks see heavy use for 
classification. So is there a Bayesian interpretation of classification? I don't know, does it?

What differs is the likelihood which in the case of classification is assumed to be 
$y \vert x, \theta \sim \mathcal{C}(\pi \vert x, \theta)$
```math
\pi = \text{Softmax}(f_\theta(x)) = \frac{\exp(f_\theta(x))}{\sum_{i=1}^K (\exp f\theta(x_i))}
```
maps the model output onto the $K$-simplex. Our likelihood function is now given by
```math
p(y \vert x \theta) = \prod_{i=1}^K y_i^{\alpha_i-1}
```

We typically minimise the Cross-Entropy
```math
H(p,q) = -\mathbb{E}_p(\log q)
```
which when $p = q = \mathcal{C}$ is

```math
H(p,q) = -\sum_{i=i}^K p(\pi_i) \log q(\pi_i) 
```
"""

# ╔═╡ e4d8c4f6-6f04-11eb-2396-ede262fdb6f3
md"""

How can we determine \(\sigma^2\) and \(\sigma^2_0\)? We can of course specify these values based on domain knowledge, but what domain knowledge do we really have regarding the weight space of models such as neural network? In these cases it might be difficult to assign sensible values, and we might instead want to leverage empirical Bayes. We can fit a model with out best guesses for both \(\sigma^2\) and \(\sigma^2_0\) and then use the residuals to estimate \(\sigma^2\) and the distribution over fit weights to estimate \(\sigma^2_0\). This process can be iterated. By doing this we are clearly running into issues with overfitting but it is a way to get parameters we can interpret.
"""

# ╔═╡ b33a7252-7123-11eb-19a4-bbdd32955f2d
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
	
	Data = ingredients("../src/data.jl")
	GR = ingredients("../src/gaussian_regression.jl")
end

# ╔═╡ c72a0ccc-711d-11eb-1be5-bf22fa41f863
begin
	using Random, Distributions, LinearAlgebra, Parameters, Plots, PlutoUI
	using Zygote: gradient
	Random.seed!(1231245)
	
	X, y = Data.linreg_toy_data()
	D, N = size(X)
	xlim = (-10, 10)
	scatter(X', y, title = "Observed data", 
		xlabel = "X", ylabel = "y",
		label = nothing
	)
end

# ╔═╡ d9caa89e-6f0d-11eb-2242-47baa342e9a9
begin
	using Printf
	prob_xx = -1:0.01:1
	pθ = Normal(0, input_σ₀^2)
	py = Normal(0, input_σ^2)
	α = input_σ^2/input_σ₀^2
	prior_plt = plot(x -> pdf(pθ, x), prob_xx, label = nothing,
		ylabel = "p(θ)", xlabel = "θ", title = "Prior", ylim = (0, 2))
	annotate!(prior_plt, [(0.8, 1.7, Plots.text("α = $(round(α; digits=4))", 16))])
	plot(prior_plt)	
end

# ╔═╡ 0fe3637e-73c0-11eb-30c0-9daf35a802cf
begin
	for i in 1:10
		L = marginal_likelihood(
			Matrix(X'), y, zeros(1), 
			diagm(ones(1)), 0.1.*I(N), ϕnn
		)
		L(Ws, bs) = begin
			Σ = diagm(ones(1))
			Λ = 0.1.*I(N)
			μ = zeros(1)
			ϕX = ϕnn(X', Ws, bs)
			κXX = ϕX'*Σ*ϕX
			# @show size(κXX)
			r = y .- ϕX'*μ
			a = κXX + Λ
			loss = r'*inv(a)*r + log(det(a))
			e = loss #log(det(κXX + Λ))
			@show e
			e
		end
		
		∇Ws, ∇bs = gradient(L, Ws, bs)
		@show ∇Ws, ∇bs  
		for (W, b, ∇W, ∇b) in zip(Ws, bs, ∇Ws, ∇bs)
			W .-= 0.1*∇W
			b .-= 0.1*∇b
		end
	end
end

# ╔═╡ aba3a4be-7450-11eb-2a00-6581be94dcec
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
	
	relu(θ) = x -> max.(0., x' .- θ)'

	function nn(f, Ws)
		x -> begin
			for W in Ws
				x = f(W)(x)
			end
			x
		end
	end
	
	K = 10
	σ = 0.5
	μ = zeros(K)
	Σ = diagm(ones(K))
	ϕ = nn
	θ₀ = [
		collect(range(-10, 10, length = K))',
		collect(range(-10, 10, length = K))',
		collect(range(-10, 10, length = K))',
	]
	θ̂, losses = fit(
		θ -> -GR.log_evidence(GR.GaussianRegression(X, y, μ, Σ, σ, ϕ(relu, θ))), 
		θ₀; steps = 100
	)
	plot(losses)
end

# ╔═╡ 2f28b3e6-7466-11eb-1427-c94e7d886152
begin
	reg = GR.GaussianRegression(X, y, μ, Σ, σ, ϕ(relu, θ̂))
	xx = collect(range(-8, 8, length = 500))'
	
	pf₀ = GR.prior_f(reg, xx)
	prior_nn_plt = plot(xx', pf₀.μ, ribbon = 2*sqrt.(diag(pf₀.Σ)), 
		label = "p(f)", legend =:topleft,
		xlim = xlim, ylim = (-100, 100))
	
	for (i, f) in enumerate(eachcol(rand(pf₀, 5)))
        plot!(
            prior_nn_plt, xx', f, color=3,
            label=i == 1 ? "Prior sample" : nothing
        )
    end
	
	scatter!(prior_nn_plt, X', y, label = "Observations",
		color = 1,
		title ="Prior predictive",
		xlabel = "x", ylabel = "f(x)")
	
	pf = GR.posterior_f(reg, xx)
	posterior_nn_plt = plot(xx', pf.μ, ribbon = 2*sqrt.(diag(pf.Σ)), 
		color = 1, label = "p(f | X, y)", legend =:topleft,
		xlim = xlim, ylim = (-10, 10))
	for (i, f) in enumerate(eachcol(rand(pf, 5)))
        plot!(
            posterior_nn_plt, xx', f, color=3,
            label=i == 1 ? "Posterior sample" : nothing
        )
    end
	scatter!(posterior_nn_plt, X', y, label = "Observations",
		color = 1,
		title ="Posterior predictive",
		xlabel = "x", ylabel = "f(x)")
	
	plot(prior_nn_plt, posterior_nn_plt, layout = (2,1), size = (660, 600))
end

# ╔═╡ b8e5bd12-75cf-11eb-19df-379c49419514
md"""
Log evidence: $(GR.log_evidence(reg))
"""

# ╔═╡ Cell order:
# ╠═a2ac2eac-7111-11eb-2101-b198c13d3f5f
# ╠═c72a0ccc-711d-11eb-1be5-bf22fa41f863
# ╠═aba3a4be-7450-11eb-2a00-6581be94dcec
# ╟─b8e5bd12-75cf-11eb-19df-379c49419514
# ╠═2f28b3e6-7466-11eb-1427-c94e7d886152
# ╠═f88bc10a-75cf-11eb-001b-35d72fde997b
# ╠═4b90cb7e-6f00-11eb-2fe7-af7d99ad7ed9
# ╠═d9caa89e-6f0d-11eb-2242-47baa342e9a9
# ╠═278e0f08-6fbd-11eb-25e8-5b73d8d7ed6d
# ╠═e4d8c4f6-6f04-11eb-2396-ede262fdb6f3
# ╠═0fe3637e-73c0-11eb-30c0-9daf35a802cf
# ╠═b33a7252-7123-11eb-19a4-bbdd32955f2d
