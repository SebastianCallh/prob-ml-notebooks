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

 	GP = ingredients("../src/gaussian_process.jl")
 	Data = ingredients("../src/data.jl")
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

# ╔═╡ 164a8246-712d-11eb-29a8-654f3e2d6f72
md"""
# Unlimited features!

[Probabilistic ML - Lecture 9 - Gaussian Processes](https://www.youtube.com/watch?v=s2_L86D4kUE&list=PL05umP7R6ij1tHaOFY96m5uX3J21a6yNd&index=10)
"""

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
The choice of kernel greatly affects the kind of data we can observe.

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
Evidence: $(GP.log_evidence(gp, X, y))
"""

# ╔═╡ Cell order:
# ╠═f3922600-712c-11eb-2bca-db4ed5d1552a
# ╠═164a8246-712d-11eb-29a8-654f3e2d6f72
# ╠═792a5b4e-712c-11eb-23f6-895892faac22
# ╠═e8ed0dcc-714b-11eb-3841-59355f0e8d89
# ╠═1de6f408-714b-11eb-21ba-3fac6f051fc2
# ╠═69a2ae8e-715d-11eb-3e9b-cbdd846616ee
# ╠═42ef5752-712d-11eb-1465-4d5254770d45
