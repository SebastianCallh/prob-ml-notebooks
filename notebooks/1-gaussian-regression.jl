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
	using Random, Distributions, LinearAlgebra, Plots, PlutoUI
	Random.seed!(1231245)
	# plotly()
	
	N = 50
	X = collect(range(-6, 6; length = N)) .+ randn(N)
	y = X .+ randn(N)
	y[10:30] .= y[10:30] .+ 4
	y[30:40] .= sin.(y[30:40]) 
	scatter(X, y, title = "Observed data", 
		label = nothing, xlabel = "X", ylabel = "y")
end

# ╔═╡ 67c6886a-6a4b-11eb-377b-278d91b6d03f


# ╔═╡ b6cfc0c6-6dfb-11eb-058d-db10ed33d71a
md"""
## Gaussian regression
Probabilistic inference is generally computationally difficult. But if we can assume that our quantities of interest follow a Gaussian distribution, inference reduces to linear algebra operations. In this notebook we are going to look at how to perform regression using this framework. Since there is a quite a bit of maths involved, we will start with a simple linear regression to not introduce too many moving pieces. Our data is given by the data matrix $X \in \mathbb{R}^{D \times N}$ with $N$ observations of dimension $D$ and the observed values $\mathbf{y} \in \mathbb{R}^N$.

[Probabilistic ML - Lecture 7 - Gaussian Parametric Regression](https://www.youtube.com/watch?v=EF1BfKnINw0&t=3210s)
"""

# ╔═╡ 79934a56-6e1c-11eb-0728-1b6a25618285
md"""
We assume the model $y(x) = \phi_x^T w$ where $\phi_x = \phi(x) : \mathbb{R}^N \mapsto \mathbb{R}^K$ is the *feature function* evaluated at $x$ and $w$ the model weights. In the case of a linear model this means $\phi_x = \left(1, x \right)^T$. We also assume the model weights \$w$ to be Gaussian with prior $p(w) = \mathcal{N}(w; \mu, \Sigma)$, and finally
we assume a Gaussian likelihood
$p(y \vert w, \phi_X) = \mathcal{N}(y; \phi_X^T w, \sigma^2I)$ with (known) independent observation noise $\sigma^2$. 

We can now compute the induced prior distribution over functions for a given $p(w)$, which is given by $p(f) = \mathcal{N}(f; \phi_x \mu, \phi_x^T \Sigma \phi_x)$.
To get a feeling for how these quantities interact you can play around with the sliders below to assign different prior weights, and see how the distributions over functions changes.

w₁ $(@bind μw₁ Slider(-10:10; default=0, show_value=true))
σ₁ $(@bind σw₁ Slider(0.5:0.5:10; default=1, show_value=true))

w₂ $(@bind μw₂ Slider(-10:10; default=0, show_value=true))
σ₂ $(@bind σw₂ Slider(0.5:0.5:10; default=1, show_value=true))
"""

# ╔═╡ 5114873e-6dfc-11eb-0da3-21a6186d9da2
begin
	ϕ(x) = [ones(length(x)) x]'
	K = length(ϕ([1]))
	μ₀ = [μw₁, μw₂]
	Σ₀ = [σw₁ 0; 0 σw₂]
	pw = MvNormal(μ₀, Σ₀)
	xx = collect(-8:0.2:8)
	ϕₓ = ϕ(xx)
	μf = ϕₓ'*μ₀
	Σf = ϕₓ'*Σ₀*ϕₓ
	stdf = sqrt.(diag(Σf))
	ws = rand(pw, 5)
	fs = ϕₓ'*ws
	
	w1_grid = range(μ₀[1] .- 3*Σ₀[1,1], μ₀[1] .+ 3*Σ₀[1,1]; length = 100)
	w2_grid = range(μ₀[2] .- 3*Σ₀[2,2], μ₀[2] .+ 3*Σ₀[2,2]; length = 100) 
	
	pw_plt = contour(
		w1_grid, w2_grid, (x, y) -> pdf(pw, [x, y]), 
		title = "p(w)", xlabel = "w₁", ylabel = "w₂",
		xlim=extrema(w1_grid),
		ylim=extrema(w2_grid),
	)
	
	p = plot(xx, μf, ribbon=2*stdf, 
		title = "p(f)",
		xlabel = "x",
		ylabel = "f(x)", 
		label = nothing
	)
	for (i, f) in enumerate(eachcol(fs))
		plot!(p, xx, f, color = 3,
			label = i == 1 ? "Prior sample" : nothing)
	end
	
	scatter!(p, X, y, label = "Observations", color = 1, legend =:topleft)
	plot(pw_plt, p, size = (680, 350))
end

# ╔═╡ e3c39cb8-6e11-11eb-1931-eb71caa59a07
md"""
## Gaussian Inference
To do inference we have to compute the posterior (of course!). Since we assumed everything to be Gaussian, this reduces to a bunch of linear algebra, and we can compute the posterior, which is also Gaussian as
```math
\begin{equation}
\begin{split}
p(w \vert \mathbf{y}, \phi_X) = \mathcal{N}(w; 
& \mu + \Sigma\phi_X(\phi_X^T \Sigma \phi_X + \sigma^2 I)^{-1} (\mathbf{y} - \phi_X^T \mu), \\
& \Sigma - \Sigma \phi_X (\phi_X^T \Sigma \phi_X + \sigma^2 I)^{-1}\phi_X^T \Sigma).
\end{split}
\end{equation}
```
The posterior distribution over functions is computed the same way as for the prior. That is, we multiply the features of the test points $\phi_x$ onto the mean and onto the covariance
```math
\begin{equation}
\begin{split}
p(f_x \vert w, \phi_X) = \mathcal{N}(f_x; 
& \phi_x^T \mu + \phi_x^T \Sigma\phi_X(\phi_X^T \Sigma \phi_X + \sigma^2 I)^{-1} (\mathbf{y} - \phi_X^T \mu), \\
& \phi_x^T\Sigma\phi_x - \phi_x^T \Sigma \phi_X (\phi_X^T \Sigma \phi_X + \sigma^2 I)^{-1}\phi_X^T \Sigma \phi_x.
\end{split}
\end{equation}
```

The above expressions may look a bit overwhelming, but at the end of the day they are just linear algebra. In fact, what we have here is a rare unicorn of a posterior which we are able to write down an expression for. Even though we have to multiply a bunch of vectors and invert some matrices to get it, it is very cool that it is possible at all.

#### Alterative parametrisation
We can also invoke the [matrix inversion lemma](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) and express the posterior over weights as
```math
\begin{equation}
\begin{split}
p(w \vert \mathbf{y}, \phi_X) = \mathcal{N}(w; 
& \left( \Sigma^{-1} + \sigma^{-2} \phi_X \phi_X^T \right)^{-1} 
\left (\Sigma^{-1} \mu + \sigma^{-2} \phi_X \mathbf{y} \right), \\
&\left (\Sigma^{-1} + \sigma^{-2} \phi_X \phi_X^T \right)^{-1}),
\end{split}
\end{equation}
```
and over function values as
```math
\begin{equation}
\begin{split}
p(f_x \vert \mathbf{y}, \phi_X) = \mathcal{N}(w; 
& \phi_x^T\left( \Sigma^{-1} + \sigma^{-2} \phi_X \phi_X^T \right)^{-1} 
\left (\Sigma^{-1} \mu + \sigma^{-2} \phi_X \mathbf{y} \right), \\
& \phi_x^T \left (\Sigma^{-1} + \sigma^{-2} \phi_X \phi_X^T \right)^{-1} \phi_x).
\end{split}
\end{equation}
```
This is useful since $\phi_X^T \Sigma \phi_X$ is $N \times N$ while 
$\Sigma^{-1} + \sigma^{-2} \phi_X^T \phi_X$ is $K \times  K$. When the number of  features is smaller than the number of observation (which is not unlikely) the second parametrisation is more efficient. Finally, to clear up notation, let us denote the inner products and the residual between the observations and prior predictions as

```math
\begin{equation}
\begin{split}
\kappa_{ab} = & \phi_a^T \Sigma\phi_b \\
\mathbf{r}  = & \mathbf{y} - \phi_X^T \mu
\end{split}
\end{equation}
```

which lets us express the posterior over functions as 
```math
\begin{equation}
\begin{split}
p(f_x \vert w, \phi_X) = \mathcal{N}(f_x; 
& \phi_x^T \mu + \kappa_{xX}(\kappa_{XX} + \sigma^2 I)^{-1} \mathbf{r}, \\
& \kappa_{xx} - \kappa_{xX} (\kappa_{XX} + \sigma^2 I)^{-1}\kappa_{Xx}.
\end{split}
\end{equation}
```
Let us now use identities to compute the posterior.
"""

# ╔═╡ 6f01a140-6ef2-11eb-3bf1-cb378acd4bab
begin
	function posterior(X, y, μ = ones(2), Σ = I(2), σ = 0.5)
		ϕX = ϕ(X)
		ϕx = ϕ(xx)
		κxX = ϕx'*Σ*ϕX
		κXX = ϕX'*Σ*ϕX
		κxx = ϕx'*Σ*ϕx
		r = y - ϕX'*μ
		A = κXX + σ^2*I(N)
		G = cholesky(A)
		A = (G\κxX')'
		μₙ = ϕx'*μ + A*r
		Σₙ = κxx - A*κxX'
		μₙ, Σₙ
	end
	
	function posterior_alt(X, y, μ = ones(2), Σ = I(2), σ = 0.5) 
		ϕX = ϕ(X)
		ϕx = ϕ(xx)
		invΣ = inv(Σ)
		A = invΣ + σ^(-2)*ϕX*ϕX'
		G = cholesky(A)
		A = (G\ϕx)'
		μₙ = A*(invΣ*μ + σ^(-2)*ϕX*y)
		Σₙ = A*ϕx
		μₙ, Σₙ
	end
end;

# ╔═╡ 2d893066-6e1b-11eb-147a-8f5062800b8d
begin
	μₙ, Σₙ = posterior(X, y) 
	alt_μₙ, alt_Σₙ = posterior_alt(X, y) 
	
	posterior_plt = scatter(
		X, y, label = "Observations", 
		title ="Posterior predictive",
		xlabel = "x", ylabel = "f(x)"
	)
	plot!(posterior_plt , xx, μₙ, 
		ribbon = 2*sqrt.(diag(Σₙ)), label = "p(f | X, y)",
		legend = :topleft
	)
	plot!(posterior_plt , xx, alt_μₙ, 
		ribbon = 2*sqrt.(diag(alt_Σₙ)), label = "p(f | X, y) alt.",
		legend = :topleft
	)
end

# ╔═╡ 0db1d1bc-6e3c-11eb-0b32-b9a56eb0b2e6
md"""
The plot confirms that both identities produce the same posterior. And while the posterior makes sense, it does not capture the data particularly well. Additionally, it is very(!) overconfident. There is clearly room for improvement in our modelling.
	
### Ending notes
We have seen how to infer model weights $w$ in closed form using Gaussian distributions. We saw two identities for this, one that is more efficient when the number of parameters is larger than the number of observations, and one that is more efficient when the opposite is true.

We also saw how the model is surprisingly overconfident. This can be explained by the fact that the posterior uncertainty does not depend on observations $y$, but only on $x$. Theere is simply no information about predictions being far away from observed values that can be taken into account. While we used a linear regression, the only requirement we had was that the model was linear *in the weights*. In the next notebook we are going to make use of this to greatly improve the model fit.
"""

# ╔═╡ Cell order:
# ╟─67c6886a-6a4b-11eb-377b-278d91b6d03f
# ╠═b6cfc0c6-6dfb-11eb-058d-db10ed33d71a
# ╠═23451f6e-6e1d-11eb-35fb-1b560a12e694
# ╠═79934a56-6e1c-11eb-0728-1b6a25618285
# ╠═5114873e-6dfc-11eb-0da3-21a6186d9da2
# ╟─e3c39cb8-6e11-11eb-1931-eb71caa59a07
# ╠═6f01a140-6ef2-11eb-3bf1-cb378acd4bab
# ╠═2d893066-6e1b-11eb-147a-8f5062800b8d
# ╠═0db1d1bc-6e3c-11eb-0b32-b9a56eb0b2e6
