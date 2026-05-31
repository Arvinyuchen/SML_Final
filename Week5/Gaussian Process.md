## Recap: From Linear Regression to Kernelised Form

**Setup.** Let $D$ = number of features, $N$ = number of training points, $\boldsymbol{\Phi} \in \mathbb{R}^{N \times D}$ the feature (design) matrix, and $\mathbf{y} \in \mathbb{R}^N$ the target vector.

**Regularised linear regression (Week 2a).** The prediction and optimal weights are
$$
f(\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^\top \mathbf{w}^*, \qquad \mathbf{w}^* = (\lambda \mathbf{I}_D + \boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}.
$$
The matrix inverse here is $D \times D$ — we work in **feature space**.

**Kernelised form (Week 4a, Bishop 6.9).** Using the identity from Week 4, we can rewrite the prediction entirely in terms of the kernel:
$$
f(\mathbf{x}) = \mathbf{k}(\mathbf{x})^\top (\lambda \mathbf{I}_N + \mathbf{K})^{-1} \mathbf{y}, \tag{6.9}
$$
where $\mathbf{K} \in \mathbb{R}^{N \times N}$ is the **Gram matrix** with entries $K_{mn} = k(\mathbf{x}_m, \mathbf{x}_n) = (\boldsymbol{\Phi}\boldsymbol{\Phi}^\top)_{mn}$, and $\mathbf{k}(\mathbf{x})$ is the vector of kernel evaluations against all training points. The Gram matrix is positive semi-definite (Bishop 6.10, 6.11). The inverse is now $N \times N$ — we work in **kernel space**.

### Why Kernels?

Kernels simplify the process of coming up with features. Instead of designing $\boldsymbol{\phi}(\mathbf{x})$ explicitly, we specify a kernel $k(\mathbf{x}_m, \mathbf{x}_n)$ that implicitly defines the feature space:

| Kernel | Implicit feature map |
|---|---|
| $k(\mathbf{x}_m, \mathbf{x}_n) = \mathbf{x}_m^\top \mathbf{x}_n$ (linear) | $\boldsymbol{\phi}(\mathbf{x}) = \mathbf{x}$ |
| $k(\mathbf{x}_m, \mathbf{x}_n) = (\mathbf{x}_m^\top \mathbf{x}_n)^2$ (quadratic, Bishop 6.12) | $\boldsymbol{\phi}(\mathbf{x}) = (x_1^2, \sqrt{2}\,x_1 x_2, x_2^2)^\top$ in 2D |
| $k(\mathbf{x}_m, \mathbf{x}_n) = \exp\!\left(-\frac{\|\mathbf{x}_n - \mathbf{x}_m\|^2}{2\sigma^2}\right)$ (RBF, Bishop 6.23) | **Infinite**-dimensional feature space |

The RBF kernel gives access to an infinite-dimensional feature space without ever computing $\boldsymbol{\phi}$ — this is the power of the kernel trick.

## Recap: Bayesian Linear Regression

### The Probabilistic Setup (Week 2b)

Instead of finding a single point estimate $\mathbf{w}^*$, we place a distribution over $\mathbf{w}$ and update it with data.

**Likelihood (Bishop 3.10).** Each observation is the linear prediction plus Gaussian noise with precision $\beta$:
$$
p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) = \prod_{n=1}^{N} \mathcal{N}(y_n;\; \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_n),\; \beta^{-1}).
$$

**Conjugate prior (Bishop 3.52).** A zero-mean isotropic Gaussian over the weights:
$$
p(\mathbf{w}) = \mathcal{N}(\mathbf{w};\; \mathbf{0},\; \lambda^{-1} \mathbf{I}_D).
$$
The prior precision $\lambda$ plays the same role as the regularisation parameter — larger $\lambda$ means stronger shrinkage toward zero.

### The Posterior (Bishop 3.49, 3.53–3.54)

Since the likelihood is Gaussian and the prior is conjugate Gaussian, the posterior is also Gaussian:
$$
p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) = \mathcal{N}(\mathbf{w};\; \boldsymbol{\mu},\; \boldsymbol{\Sigma}),
$$
with
$$
\boldsymbol{\Sigma}^{-1} = \lambda \mathbf{I}_D + \beta\, \boldsymbol{\Phi}^\top \boldsymbol{\Phi}, \tag{3.54}
$$
$$
\boldsymbol{\mu} = \beta\, \boldsymbol{\Sigma}\, \boldsymbol{\Phi}^\top \mathbf{y} = \beta\, (\lambda \mathbf{I}_D + \beta\, \boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}. \tag{3.53}
$$

Note: when $\beta = 1$, the posterior mean $\boldsymbol{\mu}$ recovers the regularised least-squares solution $\mathbf{w}^*$ exactly. The Bayesian treatment gives us the same point prediction *plus* a full covariance $\boldsymbol{\Sigma}$ quantifying uncertainty over $\mathbf{w}$ — this is the $D \times D$ computation that will be kernelised next to obtain the Gaussian process.

