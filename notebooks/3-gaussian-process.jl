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

# ╔═╡ 164a8246-712d-11eb-29a8-654f3e2d6f72
md"""
# Gaussian processes
We have previously seen how we can stack feature learners to learn more abstract features and create deep neural networks. In this notebook we are instead going to consider using an infinite amount of features. This *sounds* impossible, but it is in fact possible to construct a finite representation of an infinite amount of features throught the [kernel trick](https://en.wikipedia.org/wiki/Kernel_method).

[Probabilistic ML - Lecture 9 - Gaussian Processes](https://www.youtube.com/watch?v=s2_L86D4kUE&list=PL05umP7R6ij1tHaOFY96m5uX3J21a6yNd&index=10)
"""

# ╔═╡ 8a63acf4-74f0-11eb-017b-a9cff89c7416
md"""
Here's our data again.
"""

# ╔═╡ 946d3c80-74f0-11eb-2529-6950c042cb57
md"""
### Infinitely many features
Let us see what happens when we distribute the features evenly over the input space and allow the number of features $F$ to go to infinity, or equivalently the distance between them to go to zero. For simplicity let us assume 
$\Sigma = \frac{\sigma^2 (c_\text{max} - c_\text{min}))}{F}I$ is a diagonal matrix where $c_\text{max}$ and $c_\text{min}$ denote the "leftmost" and "rightmost" feature and $F$ denotes the number of features. With this covariance matrix we can write

```math
\begin{equation}
\begin{split}
\phi(x_i) \Sigma \phi(x_j) = 
\frac{\sigma^2 (c_\text{max} - c_\text{min}))}{F}I \sum_{\ell = 1}^F \phi_\ell(x_i)\phi_\ell(x_j). 
\end{split}
\end{equation}
```
If we now consider Gaussian features

```math
\begin{equation}
\begin{split}
\phi_\ell(x) = \exp \left ( - \frac{(x - c_\ell)^2}{2\lambda^2} \right)
\end{split}
\end{equation}
```
we get 
```math
\begin{equation}
\begin{split}
\phi(x_i) \Sigma \phi(x_j)
& = \frac{\sigma^2 (c_\text{max} - c_\text{min}))}{F} \sum_{\ell = 1}^F \phi_\ell(x_i)\phi_\ell(x_j) \\
& = \frac{\sigma^2 (c_\text{max} - c_\text{min}))}{F} \sum_{\ell = 1}^F 
\exp \left ( - \frac{(x_i - c_\ell)^2}{2\lambda^2} \right) 
\exp \left ( - \frac{(x_j - c_\ell)^2}{2\lambda^2} \right) \\
& = \frac{\sigma^2 (c_\text{max} - c_\text{min}))}{F}
\exp \left ( - \frac{(x_i - x_j)^2}{4\lambda} \right )
\sum_{\ell = 1}^F
\exp \left ( - \frac{(c_\ell - \frac{1}{2}(x_i + x_j))^2}{\lambda^2} \right).
\end{split}
\end{equation}
```

We are now interested in what happens when $F \to \infty$. It turns out the sum over the features turns into a Riemann integral

```math
\begin{equation}
\begin{split}
\phi(x_i) \Sigma \phi(x_j) = \sigma^2
\exp \left ( - \frac{(x_i - x_j)^2}{4\lambda} \right )
\int_{c_\text{min}}^{c_\text{max}} \exp \left ( - \frac{(c_\ell - \frac{1}{2}(x_i + x_j))^2}{\lambda^2} \right) dc
\end{split}
\end{equation}
```

which is a Gaussian integral with closed form solution when $c_\text{min} \to -\infty$, $c_\text{max} \to \infty$

```math
\begin{equation}
\begin{split}
\int_{c_\text{min}}^{c_\text{max}} \exp \left ( - \frac{(c_\ell - \frac{1}{2}(x_i + x_j))^2}{\lambda^2} \right) dc = \sqrt{2\pi}\lambda 
\end{split}
\end{equation}
```

and so we can express this *infinite sum* of features as 

```math
\begin{equation}
\begin{split}
\phi(x_i) \Sigma \phi(x_j) = \sqrt{2\pi}\lambda \sigma^2 \exp \left ( - \frac{(x_i - x_j)^2}{4\lambda} \right ).
\end{split}
\end{equation}
```

"""

# ╔═╡ f3922600-712c-11eb-2bca-db4ed5d1552a
 begin
 	"""This is just to include external files. Please ignore it"""
	function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
 	#  but with the modification that it returns the module instead of the last object
 	name = Symbol(basename(path))
 	m = Module(name)
 	Core.eval(
		m,
		Expr(:toplevel,
 	        :(eval(x) = $(Expr(:core, :eval))($name, x)),
			:(include(x) = $(Expr(:top, :include))($name, x)),
 	        :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, name, x)),  
 	        :(include($path))))
 		m
 	end

 	Data = ingredients("../src/data.jl")
 	GP = ingredients("../src/gaussian_process.jl")
end

# ╔═╡ 792a5b4e-712c-11eb-23f6-895892faac22
begin
	using Random, Distributions, LinearAlgebra, Parameters, Plots, PlutoUI
	using Zygote: gradient

	X, y = Data.linreg_toy_data()
	y = y
	D, N = size(X)
	xlim = -7, 7
	ylim = -5, 5
	σ = 0.3
	scatter(
		X', y, title="Observed data",
		xlabel="X", ylabel="y",
		label=nothing
	)
end

# ╔═╡ e8ed0dcc-714b-11eb-3841-59355f0e8d89
kernels = Dict(
	"linear" => GP.linear(zeros(size(X, 1))),
	"rq" => GP.rq(1, 1),
	"periodic" => GP.periodic(1, 1),
	"rbf" => GP.rbf(4),
);

# ╔═╡ 1de6f408-714b-11eb-21ba-3fac6f051fc2
md"""

### Kernel functions
The choice of kernel greatly affects the kind of data we can model. Different kernels correspond to different feature spaces, which have different inner products, which dictate which observations are considered close.

kernel $(@bind kernel1 Select(collect(keys(kernels))))

"""

# ╔═╡ 42ef5752-712d-11eb-1465-4d5254770d45
begin
	gp = GP.GaussianProcess(X, y, GP.constant(0), kernels[kernel1], σ)
	xx = collect(range(xlim..., length=500))'
	prior = GP.prior(gp, xx)
	
	n_samples = 5
	sample_labels = reshape([i == 1 ? "Sample" : false for i in 1:n_samples], 1, :)
	
	prior_p = plot(xlim = xlim, ylim = (-10, 10),
		xlabel = "x", ylabel = "y",
		title = "Prior predictive",
		legend = :topleft
	)
	
	plot!(
		prior_p, xx[:], prior.μ,
		ribbon = 2*sqrt.(diag(prior.Σ) .+ σ^2), 
		label = "p(f)"
	)
	plot!(prior_p, xx[:], rand(prior, n_samples), label = sample_labels, color = 3)
	scatter!(prior_p, X[:], y, label = "Observations", color = 1)
	
	posterior = GP.posterior(gp, xx)
	posterior_p = plot(xlim = xlim, ylim = (-20, 20))
	plot!(
		posterior_p, xx[:], posterior.μ,
		ribbon = 2*sqrt.(diag(prior.Σ) .+ σ^2),
		label = "p(f | X, y)",
		title = "Posterior predictive",
		xlabel = "x",
		ylabel = "f(x)",
		legend = :topleft
	)
	
	plot!(posterior_p, xx[:], rand(posterior, n_samples),
		label = sample_labels, color = 3)
	scatter!(posterior_p, X[:], y, label = "Observations", color = 1)
	
	plot(prior_p, posterior_p, layout = (2, 1), size = (670, 700))
end

# ╔═╡ 69a2ae8e-715d-11eb-3e9b-cbdd846616ee
md"""
Log evidence: $(GP.log_evidence(gp))
"""

# ╔═╡ Cell order:
# ╠═164a8246-712d-11eb-29a8-654f3e2d6f72
# ╠═8a63acf4-74f0-11eb-017b-a9cff89c7416
# ╠═792a5b4e-712c-11eb-23f6-895892faac22
# ╠═946d3c80-74f0-11eb-2529-6950c042cb57
# ╟─e8ed0dcc-714b-11eb-3841-59355f0e8d89
# ╠═1de6f408-714b-11eb-21ba-3fac6f051fc2
# ╠═69a2ae8e-715d-11eb-3e9b-cbdd846616ee
# ╠═42ef5752-712d-11eb-1465-4d5254770d45
# ╟─f3922600-712c-11eb-2bca-db4ed5d1552a
