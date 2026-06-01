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

### The Predictive Distribution (Bishop 3.58–3.59)

Given a new test input $\mathbf{x}_*$, the predictive distribution integrates over all possible weights, weighted by the posterior:
$$
p(y_* \mid \mathbf{x}_*, \mathbf{X}, \mathbf{y}) = \int p(y_* \mid \mathbf{x}_*, \mathbf{w})\, p(\mathbf{w} \mid \mathbf{X}, \mathbf{y})\, d\mathbf{w}.
$$
Both terms under the integral are Gaussian, so the result is Gaussian (Bishop 3.58–3.59):
$$
p(y_* \mid \mathbf{x}_*, \mathbf{X}, \mathbf{y}) = \mathcal{N}(y_*;\; \boldsymbol{\mu}^\top \boldsymbol{\phi}(\mathbf{x}_*),\; \sigma^2(\mathbf{x}_*)),
$$
where the predictive variance has two components:
$$
\sigma^2(\mathbf{x}_*) = \underbrace{\beta^{-1}}_{\text{observation noise}} + \underbrace{\boldsymbol{\phi}(\mathbf{x}_*)^\top \boldsymbol{\Sigma}\, \boldsymbol{\phi}(\mathbf{x}_*)}_{\text{uncertainty due to } \mathbf{w}}. \tag{3.59}
$$
This is the advantage of the full Bayesian treatment: not just a point prediction but a calibrated uncertainty that accounts for both noise and model uncertainty.

## Gaussian Process — Weight-Space Perspective (Bishop 6.4.1)

**Key idea:** a Gaussian process is what you get when you **kernelise Bayesian linear regression** — replace the $D \times D$ feature-space computations with $N \times N$ kernel-space computations.

### Kernelising the Predictive Mean

Start from the predictive mean $m(\mathbf{x}_*) = \boldsymbol{\phi}(\mathbf{x}_*)^\top \boldsymbol{\mu}$ and substitute $\boldsymbol{\mu} = \beta(\lambda \mathbf{I}_D + \beta \boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top \mathbf{y}$. Apply the kernel identity (from Week 4a) to push $\boldsymbol{\phi}(\mathbf{x}_*)^\top$ through:
$$
m(\mathbf{x}_*) = \beta\, \boldsymbol{\phi}(\mathbf{x}_*)^\top \boldsymbol{\Phi}^\top (\lambda \mathbf{I}_N + \beta\, \boldsymbol{\Phi}\boldsymbol{\Phi}^\top)^{-1} \mathbf{y} = \mathbf{k}(\mathbf{x}_*, \mathbf{X})\, (\mathbf{K} + \lambda \mathbf{I}_N)^{-1}\, \mathbf{y},
$$
where $\mathbf{k}(\mathbf{x}_*, \mathbf{X})$ is the row vector of kernel evaluations between $\mathbf{x}_*$ and each training point, and $\mathbf{K} = \boldsymbol{\Phi}\boldsymbol{\Phi}^\top$ (absorbing $\beta$ into the kernel/noise parameterisation).

### Kernelising the Predictive Variance

Similarly, the predictive variance $\sigma^2(\mathbf{x}_*) = \beta^{-1} + \boldsymbol{\phi}(\mathbf{x}_*)^\top \boldsymbol{\Sigma}\, \boldsymbol{\phi}(\mathbf{x}_*)$ can be rewritten via the same identity:
$$
\sigma^2(\mathbf{x}_*) = \beta^{-1} + k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}(\mathbf{x}_*, \mathbf{X})\, (\mathbf{K} + \lambda \mathbf{I}_N)^{-1}\, \mathbf{k}(\mathbf{X}, \mathbf{x}_*).
$$

### Summary: Bayesian Linear Regression vs Gaussian Process

| | Bayesian Linear Regression | Gaussian Process |
|---|---|---|
| **Predictive** | $p(y_* \mid \mathbf{x}_*, \mathbf{X}, \mathbf{y}) = \mathcal{N}(y_*;\; \boldsymbol{\mu}^\top \boldsymbol{\phi}(\mathbf{x}_*),\; \sigma^2(\mathbf{x}_*))$ | $p(y_* \mid \mathbf{x}_*, \mathbf{X}, \mathbf{y}) = \mathcal{N}(y_*;\; m(\mathbf{x}_*),\; \sigma^2(\mathbf{x}_*))$ |
| **Mean** | $\boldsymbol{\mu}^\top \boldsymbol{\phi}(\mathbf{x}_*)$ with $\boldsymbol{\mu} = \beta(\lambda \mathbf{I}_D + \beta \boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top \mathbf{y}$ | $m(\mathbf{x}_*) = \mathbf{k}(\mathbf{x}_*, \mathbf{X})(\mathbf{K} + \lambda \mathbf{I})^{-1}\mathbf{y}$ |
| **Variance** | $\beta^{-1} + \boldsymbol{\phi}(\mathbf{x}_*)^\top \boldsymbol{\Sigma}\, \boldsymbol{\phi}(\mathbf{x}_*)$ | $\beta^{-1} + k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}(\mathbf{x}_*, \mathbf{X})(\mathbf{K} + \lambda \mathbf{I})^{-1}\mathbf{k}(\mathbf{X}, \mathbf{x}_*)$ |
| **Inversion** | $D \times D$ (feature space) | $N \times N$ (kernel space) |

The GP formulation depends on the data only through kernels — the feature map $\boldsymbol{\phi}$ never appears.

## Gaussian Process — Function-Space Perspective (Bishop 6.4.1–6.4.2)

### Definition

A **Gaussian Process** is a probability distribution over functions $f(\mathbf{x})$ such that for any finite collection of inputs $(\mathbf{x}_1, \dots, \mathbf{x}_n)$, the joint distribution over function values is multivariate Gaussian:
$$
p(f(\mathbf{x}_1), \dots, f(\mathbf{x}_n)) = \mathcal{N}(\boldsymbol{\mu}, \mathbf{K}),
$$
where $\mu_i = m(\mathbf{x}_i)$ and $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$. The mean function is commonly set to zero (no prior knowledge about $f$), so a GP is **fully specified by its kernel** $k$.

### Deriving the GP Prior (Bishop 6.51–6.54)

Start from the weight-space model $\mathbf{y} = \boldsymbol{\Phi}\mathbf{w}$ with prior $p(\mathbf{w}) = \mathcal{N}(\mathbf{w};\; \mathbf{0},\; \alpha^{-1}\mathbf{I}_D)$. Since $\mathbf{y}$ is a linear transformation of a Gaussian, it is itself Gaussian. Its statistics are (Bishop 6.52–6.53):
$$
\mathbb{E}[\mathbf{y}] = \boldsymbol{\Phi}\,\mathbb{E}[\mathbf{w}] = \mathbf{0},
$$
$$
\text{cov}[\mathbf{y}] = \mathbb{E}[\mathbf{y}\mathbf{y}^\top] = \boldsymbol{\Phi}\,\mathbb{E}[\mathbf{w}\mathbf{w}^\top]\,\boldsymbol{\Phi}^\top = \alpha^{-1}\boldsymbol{\Phi}\boldsymbol{\Phi}^\top = \mathbf{K}, \tag{6.53}
$$
where $K_{nm} = k(\mathbf{x}_n, \mathbf{x}_m) = \alpha^{-1}\boldsymbol{\phi}(\mathbf{x}_n)^\top \boldsymbol{\phi}(\mathbf{x}_m)$. So the GP prior over function values is $p(\mathbf{y}) = \mathcal{N}(\mathbf{0}, \mathbf{K})$.

This also explains why we can work in an infinite-dimensional function space: we only ever evaluate the GP at a *finite* set of data points, and the resulting distribution is a finite-dimensional Gaussian governed by the kernel matrix.

### Adding Observation Noise (Bishop 6.57–6.62)

Observed targets are noisy: $t_n = y_n + \epsilon_n$ with $\epsilon_n \sim \mathcal{N}(0, \beta^{-1})$ independent. Since the function noise and observation noise are independent, their covariances add. The joint distribution over all $N+1$ observations (training + test) is (Bishop 6.62, 6.64):
$$
p(\mathbf{y}, y_*) = \mathcal{N}\!\left(\mathbf{0},\; \mathbf{K}_y\right), \qquad (K_y)_{mn} = k(\mathbf{x}_m, \mathbf{x}_n) + \beta^{-1}\delta_{mn}. \tag{6.62}
$$
Written in block form for training points and test point:
$$
\begin{pmatrix} \mathbf{y} \\ y_* \end{pmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\; \begin{pmatrix} \mathbf{K} + \sigma^2\mathbf{I}_N & \mathbf{k}(\mathbf{X}, \mathbf{x}_*) \\ \mathbf{k}(\mathbf{x}_*, \mathbf{X}) & k(\mathbf{x}_*, \mathbf{x}_*) + \sigma^2 \end{pmatrix}\right),
$$
where $\sigma^2 = \beta^{-1}$.

### GP Posterior: Conditioning (Bishop 6.66–6.67)

The posterior is obtained by conditioning the joint Gaussian on the observed training data, using the standard Gaussian conditioning formulae (Bishop 2.81–2.82):
$$
\boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}(\mathbf{x}_b - \boldsymbol{\mu}_b), \qquad \boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}.
$$
Applying this to the block structure above yields the **GP predictive distribution**:
$$
p(y_* \mid \mathbf{x}_*, \mathbf{X}, \mathbf{y}) = \mathcal{N}(y_*;\; m(\mathbf{x}_*),\; \sigma^2(\mathbf{x}_*)),
$$
$$
\boxed{m(\mathbf{x}_*) = \mathbf{k}(\mathbf{x}_*, \mathbf{X})\,(\mathbf{K} + \sigma^2 \mathbf{I}_N)^{-1}\,\mathbf{y}} \tag{6.66}
$$
$$
\boxed{\sigma^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) + \sigma^2 - \mathbf{k}(\mathbf{x}_*, \mathbf{X})\,(\mathbf{K} + \sigma^2 \mathbf{I}_N)^{-1}\,\mathbf{k}(\mathbf{X}, \mathbf{x}_*)} \tag{6.67}
$$

**Interpreting the predictive distribution:**
- The **mean** $m(\mathbf{x}_*)$ is a weighted combination of training targets, where the weights depend on how close $\mathbf{x}_*$ is to each training point (through the kernel). This can be written as $m(\mathbf{x}_*) = \sum_{n=1}^{N} a_n\, k(\mathbf{x}_n, \mathbf{x}_*)$ where $\mathbf{a} = (\mathbf{K} + \sigma^2 \mathbf{I})^{-1}\mathbf{y}$ (Bishop 6.68).
- The **variance** $\sigma^2(\mathbf{x}_*)$ does **not depend on the observed targets** $\mathbf{y}$ — it depends only on the input locations. Uncertainty is reduced near training points (where the kernel correlation is high) and grows in regions far from any training data.

### Visualising the GP

**Prior:** sampling from $\mathcal{N}(\mathbf{0}, \mathbf{K})$ produces random functions whose smoothness and amplitude are governed by the kernel.

**Posterior:** conditioning on data "rejects" prior functions that do not pass through (or near) the observations, leaving a narrower band of plausible functions that interpolates the data.

## Kernel Design (Bishop 6.63)

A widely used kernel for GP regression is:
$$
k(\mathbf{x}_n, \mathbf{x}_m) = \theta_0 \exp\!\left(-\frac{\theta_1}{2}\|\mathbf{x}_n - \mathbf{x}_m\|^2\right) + \theta_2 + \theta_3\, \mathbf{x}_n^\top \mathbf{x}_m. \tag{6.63}
$$

The hyperparameters control the function properties:
- $\theta_0$ — **signal amplitude** (vertical scale of the functions).
- $\theta_1$ — **inverse length-scale** (large $\theta_1$ = rapid variation / wiggly; small $\theta_1$ = smooth).
- $\theta_2$ — **constant offset** (shifts the mean level).
- $\theta_3$ — **linear trend** (adds a parametric linear component).

## Computation: The Cholesky Algorithm

The GP predictive equations require solving $(\mathbf{K} + \sigma^2 \mathbf{I}_N)^{-1}\mathbf{y}$, which involves inverting an $N \times N$ matrix. Rather than computing the inverse directly ($\mathcal{O}(N^3)$ and numerically unstable), we use the **Cholesky decomposition**:

1. **Decompose:** $\mathbf{K} + \sigma^2 \mathbf{I}_N = \mathbf{L}\mathbf{L}^\top$ where $\mathbf{L}$ is lower-triangular. Cost: $N^3/3$ operations.
2. **Solve for weights:** $\boldsymbol{\alpha} = \mathbf{L}^{\top}\backslash(\mathbf{L}\backslash\mathbf{y})$ (two triangular solves, $\mathcal{O}(N^2)$).
3. **Predictive mean:** $m(\mathbf{x}_*) = \mathbf{k}(\mathbf{x}_*, \mathbf{X}) \cdot \boldsymbol{\alpha}$.
4. **Predictive variance:** solve $\mathbf{v} = \mathbf{L}\backslash\mathbf{k}(\mathbf{X}, \mathbf{x}_*)$, then $\sigma^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{v}^\top\mathbf{v}$.

The overall complexity is $\mathcal{O}(N^3)$ for training (dominated by the Cholesky step) and $\mathcal{O}(N^2)$ per test prediction. Cholesky is preferred over LU or QR because it exploits the symmetric positive-definiteness of the kernel matrix, giving roughly $2\times$ speedup and better numerical stability.

## Learning Hyperparameters (Bishop 6.4.3, 6.69–6.70)

The kernel hyperparameters $\boldsymbol{\theta}$ (e.g. length-scale, signal variance, noise variance) are learned by maximising the **log marginal likelihood** — the probability of the observed data under the GP prior, with the function values marginalised out:
$$
\ln p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2}\ln|\hat{\mathbf{K}}| - \frac{1}{2}\mathbf{y}^\top \hat{\mathbf{K}}^{-1}\mathbf{y} - \frac{N}{2}\ln(2\pi), \tag{6.69}
$$
where $\hat{\mathbf{K}} = \mathbf{K}(\mathbf{X}, \mathbf{X}) + \sigma^2 \mathbf{I}_N$. The three terms have intuitive roles:
- $-\frac{1}{2}\mathbf{y}^\top \hat{\mathbf{K}}^{-1}\mathbf{y}$ — **data fit** (how well the model explains the targets).
- $-\frac{1}{2}\ln|\hat{\mathbf{K}}|$ — **complexity penalty** (penalises overly flexible models).
- $-\frac{N}{2}\ln(2\pi)$ — normalisation constant.

The gradient with respect to each hyperparameter $\theta_i$ is (Bishop 6.70):
$$
\frac{\partial}{\partial \theta_i}\ln p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2}\text{Tr}\!\left(\hat{\mathbf{K}}^{-1}\frac{\partial \hat{\mathbf{K}}}{\partial \theta_i}\right) + \frac{1}{2}\mathbf{y}^\top \hat{\mathbf{K}}^{-1}\frac{\partial \hat{\mathbf{K}}}{\partial \theta_i}\hat{\mathbf{K}}^{-1}\mathbf{y}. \tag{6.70}
$$
These gradients are used with standard optimisers (e.g. conjugate gradient, L-BFGS) to find optimal hyperparameters. This is a key advantage of GPs: model selection (choosing the kernel shape) is handled automatically via the marginal likelihood, rather than requiring cross-validation.

## Bayesian Linear Regression vs Gaussian Process

| | Bayesian Linear Regression | Gaussian Process |
|---|---|---|
| **When faster** | $D \ll N$ (few features, many points) | $N \ll D$ or infinite $D$ (kernels handle this) |
| **Function class** | Restricted to the assumed feature basis $\boldsymbol{\phi}$ | Prior over *all* functions (subject to kernel) |
| **Implicit assumption** | $D$ is finite and chosen | $D$ can be infinite (depends on kernel) |
| **Model selection** | Choose features + regularisation | Choose kernel + optimise hyperparameters via marginal likelihood |
| **Complexity** | $\mathcal{O}(D^3 + D^2 N)$ | $\mathcal{O}(N^3)$ |

