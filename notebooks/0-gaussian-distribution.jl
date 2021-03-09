### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 1744c326-715c-11eb-14ce-5d1a589900e5
md"""

# Introduction
Probabilistic machine learning is tricky and computationally demanding. To make inference tractable we have to make sensible design decisions when creating our models. One of the most powerful tools at our disposal to this end is the Gaussian distribution. The reason the Gaussian distribution is so useful is that it allows efficient implementation of probabilistic inference using standard basic linear algebra operations. In fact, the Gaussian distributino has a *ton* of structure to it, adn in this notebook we are going to go through them. It should mostly be thought of as a comprehensive reference you can come back to when you need to.

If you want a lecture on the topic i found [Probabilistic ML - Lecture 6 - Gaussian Distributions](https://www.youtube.com/watch?v=FIheKQ55l4c&list=PL05umP7R6ij1tHaOFY96m5uX3J21a6yNd&index=7) to be very nice.
"""

# ╔═╡ ef6ea2e2-715e-11eb-0eea-17884c509b75
md"""

## The Gaussian distribution
Please note that we are going to be talking about Gaussian *distributions* and not realisations of one. In other words, we are manipulating probability mass and not samples.

### Probability density function
The pdf of the Gaussian is given by
```math
\begin{equation}
\begin{split}
\mathcal{N}(x;\mu, \Sigma) 
& = \frac{1}{\left( 2\pi \right)^{d/2} \left \vert \Sigma \right \vert^{1/2}}
\exp \left ( - \frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu) \right ) \\
& = \frac{1}{\left( 2\pi \right)^{d/2} \left \vert \Sigma \right \vert^{1/2}}
\exp \left ( - \frac{1}{2}\text{tr} \left((x - \mu)  (x - \mu)^T  \Sigma^{-1}\right) \right ) \\
\end{split}
\end{equation}

```
where $d$ is number of dimensions. If $d = 1$ the expression simplifies to
```math
\mathcal{N}(x; \mu, \sigma) =
\frac{1}{ \sigma \sqrt{2\pi} }
\exp \left ( - \frac{1}{2} \left( \frac{(x - \mu)}{\sigma} \right )^2 \right).
```

It can also be written in terms of its natural parameters
```math
\mathcal{N}(x; \mu, \Sigma) 
= \exp \left( a + \nu^Tx - \frac{1}{2} x^T \Lambda, x \right)
= \exp \left( a + \nu^Tx - \frac{1}{2} \text{tr}(xx^T\Lambda)\right)
```
where the natural paramters are $\Lambda = \Sigma^{-1}$ and $\nu = \Lambda\mu$, the sufficient statistics are $x$, $xx^T$ and a constant $a$.

### Gaussians are symmetric
Simply put $\mathcal{N}(x; \mu, \Sigma) =  \mathcal{N}(\mu; x, \Sigma)$.

### Closed under addition
If $x \sim \mathcal{N}(\mu_x, \Sigma_x)$, $y \sim \mathcal{N}(\mu_y, \Sigma_y)$ are independent. Then $x + y \sim \mathcal{N}(\mu_z, \Sigma_z)$, where
$\mu_z = \mu_x + \mu_y$ and $\Sigma_z = \Sigma_x + \Sigma_y$.

### Closed under affine transformations
Let $x \sim \mathcal{N}(\mu, \Sigma)$, $x \in \mathbb{R}^n$, $t \in \mathbb{R}^n$ a translation and 
$A \in \mathbb{R}^{n \times n}$ a linear projection. Then $Ax + t \sim \mathcal{N}(A\mu + t, A \Sigma A^T)$.
This identity is super useful when working with linear models.

### Closed under marginalisation
Marginalisation can be seen as a special case of an affine transformation with $t = \bf{0}$ and 

```math
A = \begin{pmatrix}
  I_d      & \bf{0} \\
  \bf{0} & \bf{0}
\end{pmatrix}
```
i.e. we project onto the basis vectors. This gives
$Ax \sim \mathcal{N}(\mu_{1:d}, \Sigma_{1:d,1:d})$
where we use $1:d$ to indicate "slicing" out the first $d$ dimensions. Note that we can re-order the dimensions as we please, so the slicing can be done on arbitrary dimensions. This property implementes the sum rule of probabilities
```math
\int p(x,y) dy = \int p(y \vert x) p(x) dy = p(x).
```

Another interesting observation is that since we can marginalise through simple slicing, any Gaussian can be seen as a marginal distribution of an even larger Gaussian. In fact, *any* Gaussian distribution you may encounter can be seen as the marginal of an infinitely large Gaussian. 

### Closed under multiplication
The product of two Gaussians is another Gaussian.

```math
\begin{equation}
\begin{split}
\mathcal{N}(x; \mu_1, \Sigma_1)\mathcal{N}(x; \mu_2, \Sigma_2) 
& = \mathcal{N}(x; \mu_3, \Sigma_3) Z \\
\Sigma_3 & = (\Sigma_1^{-1} + \Sigma_2^{-1})^{-1} \\
\mu_3 & = \Sigma_3(\Sigma_1^{-1} \mu_1 + \Sigma_2^{-1} \mu_2) \\
Z & = \mathcal{N}(\mu_1; \mu_2, \Sigma_1 + \Sigma_2).
\end{split}
\end{equation}
```

### Closed under conditioning
If $x \sim \mathcal{N}(\mu_0, \Sigma_0)$ the conditional on an observation $y$ which can be written as a linear transformation of $x$ is
```math
\begin{equation}
\begin{split}
p(x \vert Ax = y) 
= \frac{p(x, y)}{p(y)}
 = \mathcal{N}(\mu, \Sigma),
\end{split}
\end{equation}
```

where

```math
\begin{equation}
\begin{split}
\mu & = \mu_0 + \Sigma_0 A^T(A \Sigma_0 A^T)^{-1}(y - A \mu_0) \\
\Sigma & = \Sigma_0 - \Sigma_0 A^T(A \Sigma_0 A^T)^{-1}A \Sigma_0
\end{split}
\end{equation}
```

This property implements the product rule of probabilities.
"""

# ╔═╡ 70d14e2c-7162-11eb-3100-31a53fdef929
md"""
## Inference
Since we are being Bayesian, inference boils down to computing the posterior through Bayes theorem. And since we can implement the sum rule and the product rule of Gaussians in linear algebra, this is also true for computing the posterior.

###  Closed under Bayes theorem
If $y = w^Tx + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ we can compute the posterior in closed form.

```math
\begin{equation}
\begin{split}
\text{Prior} \;     & p(w)             & = \mathcal{N}(\mu_0, \Sigma_0) \\
\text{Likelihood} \; & p(y \vert w, x) & = \mathcal{N}(w^T x, \sigma^2I) \\
\text{Posterior} \;  & p(w \vert y, x) & = \mathcal{N}(\mu, \Sigma)
\end{split}
\end{equation}
```
where

```math
\begin{equation}
\begin{split}
\mu    & =  \mu_0 + \Sigma_0 x (x^T \Sigma_0 x + \sigma^2I)^{-1}(y - x^T\mu_0) \\
\Sigma & = \Sigma_0 - \Sigma_0 x(x^T \Sigma_0 x + \sigma^2I)^{-1} x \Sigma_0 
\end{split}
\end{equation}
```

which requires inverting an $N \times N$ matrix.
Alternatively we can express it as
```math
\begin{equation}
\begin{split}
\mu    & = (\Sigma_0^{-1} + \sigma^{-2} x^T x)^{-1}(\Sigma_0^{-1}\mu + \sigma^{-2} x y) \\
\Sigma & = (\Sigma_0^{-1} + \sigma^{-2} x^T x)^{-1}
\end{split}
\end{equation}
```
which instead requires inverting a $D \times D$ matrix.
"""

# ╔═╡ 1f5b1c26-7a70-11eb-1748-7b0fadf75b51
md"""
## Ending notes 
In this notebook we have seen that the Gaussian distributions has a lot of nice properties which allows us to implement the sum and product rule in an efficient manner (in linear models) using linear algebra operations. We also saw that Bayesian inference can be done in closed form (again, in linear models) which we are going to make use of in the next notebook where we will use Gaussians to learn distributions over functions.
"""

# ╔═╡ Cell order:
# ╠═1744c326-715c-11eb-14ce-5d1a589900e5
# ╠═ef6ea2e2-715e-11eb-0eea-17884c509b75
# ╟─70d14e2c-7162-11eb-3100-31a53fdef929
# ╠═1f5b1c26-7a70-11eb-1748-7b0fadf75b51
