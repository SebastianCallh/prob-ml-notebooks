### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 23451f6e-6e1d-11eb-35fb-1b560a12e694
begin
	using Random, Distributions, LinearAlgebra, Plots
	Random.seed!(1231245)
	# plotly()
	
	N = 50
	X = collect(range(-6, 6; length = N)) .+ randn(N)
	y = X .+ randn(N)
	y[10:30] .= y[10:30] .+ 4
	y[30:40] .= sin.(y[30:40]) 
	scatter(X, y, title = "Observed data")
end

# ╔═╡ 67c6886a-6a4b-11eb-377b-278d91b6d03f


# ╔═╡ b6cfc0c6-6dfb-11eb-058d-db10ed33d71a
md"""
## Bayesian linear regression
Let us do some regression. As is tradidion, we will start off with linear regression. First off, let $X \in \mathbb{R}^{D \times N}$ be the data matrix with $N$ observations of dimension $D$ and $\mathbf{y} \in \mathbb{R}^N$ the observed values.

[Probabilistic ML - Lecture 7 - Gaussian Parametric Regression](https://www.youtube.com/watch?v=EF1BfKnINw0&t=3210s)
"""

# ╔═╡ 79934a56-6e1c-11eb-0728-1b6a25618285
md"""
We are going to assume the model $y(x) = \phi_x^T w$ where $\phi_x = \phi(x) : \mathbb{R}^N \mapsto \mathbb{R}^K$ is the *feature function* evaluated at $x$ and $w$ the model weights. Let us start off with a linear model, which corresponds to $\phi_x = \left(1, x \right)$. Furthermore we are going to assume the model weights \$w$ to be Gaussian distributed with prior $p(w) = \mathcal{N}(w; \mu_0, \Sigma_0)$, and finally 
we assume a Gaussian likelihood
$p(y \vert w, \phi_X) = \mathcal{N}(y; \phi_X^T w, \sigma^2I)$ with (known) independent observation noise $\sigma^2$. The induced prior distribution over functions is then $p(f) = \mathcal{N}(f; \phi_x \mu_0, \phi_x^T \Sigma_0 \phi_x)$.

"""

# ╔═╡ 5114873e-6dfc-11eb-0da3-21a6186d9da2
begin
	ϕ(x) = [ones(length(x)) x]'
	K = length(ϕ([1]))
	μ₀ = zeros(K)
	Σ₀ = diagm(ones(K))
	pw = MvNormal(μ₀, Σ₀)
	σ = 0.2
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
		xlim=span(w1_grid),
		ylim=span(w2_grid),
	)
	
	p = plot(xx, μf, ribbon=2*stdf, 
		title = "Prior hypothesis space",
		xlabel = "x",
		ylabel = "p(f)", 
		label = nothing
	)
	for (i, f) in enumerate(eachcol(fs))
		plot!(p, xx, f, color = 3,
			label = i == 1 ? "Prior sample" : nothing)
	end
	
	scatter!(p, X, y, label = "Observations", color = 1)
	plot(pw_plt, p, size = (650, 350))
end

# ╔═╡ e3c39cb8-6e11-11eb-1931-eb71caa59a07
md"""
## Inference
To do inference we have to compute the posterior (of course!). Since we assumed everything to be Gaussian, this reduces to a bunch of linear algebra, and we can compute the posterior, which is also Gaussian as
```math
\begin{equation}
\begin{split}
p(w \vert \mathbf{y}, \phi_X) = \mathcal{N}(w; 
& \mu_0 + \Sigma_0\phi_X(\phi_X^T \Sigma_0 \phi_X + \sigma^2 I)^{-1} (\mathbf{y} - \phi_X^T \mu_0), \\
& \Sigma_0 - \Sigma_0 \phi_X (\phi_X^T \Sigma_0 \phi_X + \sigma^2 I)^{-1}\phi_X^T \Sigma_0).
\end{split}
\end{equation}
```
The posterior distribution over functions is computed the same way as for the prior. That is, we multiply the features of the test points $\phi_x$ onto the mean and onto the covariance
```math
\begin{equation}
\begin{split}
p(f_x \vert w, \phi_X) = \mathcal{N}(f_x; 
& \phi_x^T \mu_0 + \phi_x^T \Sigma_0\phi_X(\phi_X^T \Sigma_0 \phi_X + \sigma^2 I)^{-1} (\mathbf{y} - \phi_X^T \mu_0), \\
& \phi_x^T\Sigma_0\phi_x - \phi_x^T \Sigma_0 \phi_X (\phi_X^T \Sigma_0 \phi_X + \sigma^2 I)^{-1}\phi_X^T \Sigma_0 \phi_x.
\end{split}
\end{equation}
```

#### Alterative parametrisation
We can also invoke the [matrix inversion lemma](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) and express the posterior over weights as
```math
\begin{equation}
\begin{split}
p(w \vert \mathbf{y}, \phi_X) = \mathcal{N}(w; 
& \left( \Sigma_0^{-1} + \sigma^{-2} \phi_X^T \phi_X \right)^{-1} 
\left (\Sigma_0^{-1} \mu_0 + \sigma^{-2} \phi_X \mathbf{y} \right), \\
&\left (\Sigma_0^{-1} + \sigma^{-2} \phi_X^T \phi_X \right)^{-1}),
\end{split}
\end{equation}
```
and over function values as
```math
\begin{equation}
\begin{split}
p(f_x \vert \mathbf{y}, \phi_X) = \mathcal{N}(w; 
& \phi_x^T\left( \Sigma_0^{-1} + \sigma^{-2} \phi_X^T \phi_X \right)^{-1} 
\left (\Sigma_0^{-1} \mu_0 + \sigma^{-2} \phi_X \mathbf{y} \right), \\
& \phi_x^T \left (\Sigma_0^{-1} + \sigma^{-2} \phi_X^T \phi_X \right)^{-1} \phi_x).
\end{split}
\end{equation}
```
This is useful since $\phi_X^T \Sigma_0 \phi_X$ is $N \times N$ while 
$\Sigma_0^{-1} + \sigma^{-2} \phi_X^T \phi_X$ is $K \times  K$. When the number of  features is smaller than the number of observation (which happen to be the case for us) the second parametrisation is more efficient. Finally, to clear up notation, let us denote the inner products and the residual as

```math
\begin{equation}
\begin{split}
\kappa_{ab} = & \phi_a^T \Sigma_0\phi_b \\
\mathbf{r}  = & \mathbf{y} - \phi_X^T \mu_0
\end{split}
\end{equation}
```

which lets us express the posterior over functions as 
```math
\begin{equation}
\begin{split}
p(f_x \vert w, \phi_X) = \mathcal{N}(f_x; 
& \phi_x^T \mu_0 + \kappa_{xX}(\kappa_{XX} + \sigma^2 I)^{-1} \mathbf{r}, \\
& \kappa_{xx} - \kappa_{xX} (\kappa_{XX} + \sigma^2 I)^{-1}\kappa_{Xx}.
\end{split}
\end{equation}
```
"""

# ╔═╡ 89a37cf0-6e19-11eb-316c-d3381d202b81
begin
	ϕX = ϕ(X)
	ϕx = ϕ(xx)
	κxX = ϕx'*Σ₀*ϕX
	κXX = ϕX'*Σ₀*ϕX
	κxx = ϕx'*Σ₀*ϕx
	r = y - ϕX'*μ₀
	A = κXX + σ^2*I(N)
	G = cholesky(A)
	A = (G\κxX')'
	μ = ϕx'*μ₀ + A*r
	Σ = κxx - A*κxX'
end;

# ╔═╡ 2d893066-6e1b-11eb-147a-8f5062800b8d
begin
	σs = sqrt.(diag(Σ))
	posterior_plt = scatter(X, y, label = "Observations", title ="Posterior predictive")
	plot!(posterior_plt , xx, μ, ribbon = 2*σs, label = "Model fit")
end

# ╔═╡ 7b406c6e-6e42-11eb-3cf3-611594f9c15e
begin
	# model evidence
	py = MvNormal(ϕX'*μ₀, Symmetric(κXX + σ^2*I(N)));
	pdf(py, y)
end

# ╔═╡ 15ee51bc-6e42-11eb-1acf-651a3f7a0c32
md"""
While we managed to fit the model to the data, it does not capture the data particularly well, as reflected in the plot and by the abysmal model evidence. Additionally, it is very(!) overconfident.  
"""

# ╔═╡ 0db1d1bc-6e3c-11eb-0b32-b9a56eb0b2e6
md"""
### Ending notes
We have seen how to infer model weights $w$ in closed form using Gaussian distributions. While we used a linear regression, the only requirement we had was that the model was linear *in the weights*.

We also saw how the model is surprisingly overconfident. This can be explained by the fact that the posterior uncertainty does not depend on observations $y$, but only on $x$. Theere is simply no information that says that the prediction is far away from the observed values that can be taken into account. Let us think about how to improve the model fit in the next notebook.
"""

# ╔═╡ Cell order:
# ╟─67c6886a-6a4b-11eb-377b-278d91b6d03f
# ╠═b6cfc0c6-6dfb-11eb-058d-db10ed33d71a
# ╠═23451f6e-6e1d-11eb-35fb-1b560a12e694
# ╠═79934a56-6e1c-11eb-0728-1b6a25618285
# ╠═5114873e-6dfc-11eb-0da3-21a6186d9da2
# ╠═e3c39cb8-6e11-11eb-1931-eb71caa59a07
# ╠═89a37cf0-6e19-11eb-316c-d3381d202b81
# ╠═2d893066-6e1b-11eb-147a-8f5062800b8d
# ╠═7b406c6e-6e42-11eb-3cf3-611594f9c15e
# ╠═15ee51bc-6e42-11eb-1acf-651a3f7a0c32
# ╠═0db1d1bc-6e3c-11eb-0b32-b9a56eb0b2e6
