## Motivation

The basic framing is **dimensionality reduction as data compression**. We start with a high-dimensional data point
$$
\mathbf{x} \in \mathbb{R}^D
$$
and map it into a smaller representation
$$
\mathbf{z} \in \mathbb{R}^M, \qquad M < D.
$$
We then decode this low-dimensional representation back into a reconstruction
$$
\tilde{\mathbf{x}} \in \mathbb{R}^D.
$$

So the pipeline is
$$
\mathbf{x}
\xrightarrow{\text{encode}}
\mathbf{z}
\xrightarrow{\text{decode}}
\tilde{\mathbf{x}}.
$$

The code $\mathbf{z}$ should capture the desirable variation in $\mathbf{x}$, while the reconstruction $\tilde{\mathbf{x}}$ should remain close to the original input. In other words, a good low-dimensional representation keeps the informative structure and loses as little as possible.

This leads to the key question:

> How do we construct the encode and decode mappings?

PCA gives the simplest answer: make both mappings linear. It chooses the linear bottleneck that loses the least information, measured through reconstruction error.

### Why Compress Data?

There are three main motivations.

**Low intrinsic dimensionality.** Data may be recorded in a high-dimensional space $\mathbb{R}^D$ even though the important variation lies close to a much lower-dimensional structure, such as a line, plane, or curved manifold. Dimensionality reduction tries to recover that simpler structure.

**Visualisation and exploratory data analysis.** High-dimensional data cannot be plotted directly. Mapping it into two or three dimensions lets us inspect clusters, trends, outliers, and other qualitative structure.

**Downstream learning.** A classifier or regression model can be trained on the compact representation $\mathbf{z}$ instead of the original $\mathbf{x}$. This can reduce noise, reduce computation, and reduce the number of features the later model has to handle.

These motivations are all different ways of saying the same thing: we want a smaller representation that still preserves the important structure in the original data. The MNIST example shows what this looks like visually.

### MNIST Example

For MNIST, each digit image has
$$
D = 784
$$
pixel dimensions. The slide compresses these images down to
$$
M = 2.
$$

Even with only two numbers per image, the resulting plot still shows meaningful structure: digit images cluster and spread according to visual differences such as slant, loop shape, and stroke style. The example shows the basic promise of dimensionality reduction: a small number of coordinates can retain a surprising amount of the important variation in the original data.

The mental hook for the next section is:

> PCA = choose the linear bottleneck that loses the least.

The next slide turns this into a formal setup using the sample covariance matrix
$$
S = \frac{1}{N}\sum_{n=1}^{N}(\mathbf{x}_n - \bar{\mathbf{x}})(\mathbf{x}_n - \bar{\mathbf{x}})^\top,
$$
a projection
$$
\mathbf{z}_n = U^\top \mathbf{x}_n,
$$
and a reconstruction
$$
\tilde{\mathbf{x}}_n = U \mathbf{z}_n.
$$

## PCA Review: Maximum Variance Projection

The PCA review begins with the maximum-variance view of PCA. Instead of starting with reconstruction error, we ask which direction makes the projected data as spread out as possible. This gives the same principal directions and leads naturally to the eigenvector equation.

The slide uses the Bishop convention where the sample mean is written explicitly inside the covariance matrix:
$$
S =
\frac{1}{N}
\sum_{n=1}^{N}
(\mathbf{x}_n - \bar{\mathbf{x}})
(\mathbf{x}_n - \bar{\mathbf{x}})^\top.
$$

So we do not assume the data has already been centred inside the notation. Instead, centring is done by subtracting the sample mean $\bar{\mathbf{x}}$.

The principal directions are eigenvectors of the covariance matrix:
$$
S\mathbf{u}_i = \lambda_i \mathbf{u}_i.
$$
Here $\mathbf{u}_i$ is a principal direction and $\lambda_i$ is the variance captured along that direction.

Before deriving why these eigenvectors appear, it helps to recall what it means to represent a vector using basis directions. PCA is not inventing a new kind of coordinate system; it is choosing a particularly useful orthonormal coordinate system for the data.

### Expanding Data in an Orthonormal Basis

Take a full orthonormal basis
$$
\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_D.
$$
Any data point can be expanded as
$$
\mathbf{x}_n =
\sum_{i=1}^{D}
(\mathbf{x}_n^\top \mathbf{u}_i)\mathbf{u}_i.
\tag{12.9}
$$

The scalar $\mathbf{x}_n^\top \mathbf{u}_i$ is the coordinate of $\mathbf{x}_n$ along direction $\mathbf{u}_i$. This works because the basis is orthonormal:
$$
\mathbf{u}_i^\top \mathbf{u}_j =
\begin{cases}
1, & i = j, \\
0, & i \neq j.
\end{cases}
$$

Each $\mathbf{u}_i$ is a direction in $\mathbb{R}^D$, and because there are exactly $D$ orthonormal directions, they form a complete coordinate system for the original space. Using all $D$ directions gives an exact reconstruction of $\mathbf{x}_n$.

PCA starts from this full basis, but then keeps only the first $M$ most important directions:
$$
\mathbf{u}_1, \dots, \mathbf{u}_M,
\qquad
M < D.
$$
The reconstruction then becomes approximate:
$$
\tilde{\mathbf{x}}_n
=
\sum_{i=1}^{M}
(\mathbf{x}_n^\top \mathbf{u}_i)\mathbf{u}_i.
$$

So the difference is:
$$
\mathbf{x}_n =
\sum_{i=1}^{D}
(\mathbf{x}_n^\top \mathbf{u}_i)\mathbf{u}_i
$$
is exact because it uses the full basis, while
$$
\tilde{\mathbf{x}}_n =
\sum_{i=1}^{M}
(\mathbf{x}_n^\top \mathbf{u}_i)\mathbf{u}_i
$$
is approximate because it discards the directions $\mathbf{u}_{M+1}, \dots, \mathbf{u}_D$. This is the compression step in PCA.

The maximum-variance question is therefore: which basis directions should be kept first if we want to preserve as much variation in the data as possible?

PCA answers this by scoring each candidate direction according to how much the data varies after being projected onto it. A direction is useful if the projected data points are spread out along that line, because this means the direction captures meaningful variation rather than nearly constant or low-information noise. This turns the basis-selection problem into a variance-maximisation problem.

### Projected Variance

Projecting $\mathbf{x}_n$ onto the first direction $\mathbf{u}_1$ gives
$$
\mathbf{x}_n^\top \mathbf{u}_1.
$$
Because the data may not have zero mean, the projected mean is
$$
\bar{\mathbf{x}}^\top \mathbf{u}_1.
$$

Therefore the variance of the projected data is
$$
\frac{1}{N}
\sum_{n=1}^{N}
\left\{
\mathbf{x}_n^\top \mathbf{u}_1
-
\bar{\mathbf{x}}^\top \mathbf{u}_1
\right\}^2.
$$

The term inside the square can be rewritten as
$$
\mathbf{x}_n^\top \mathbf{u}_1
-
\bar{\mathbf{x}}^\top \mathbf{u}_1
=
(\mathbf{x}_n - \bar{\mathbf{x}})^\top \mathbf{u}_1.
$$
So the projected variance is
$$
\frac{1}{N}
\sum_{n=1}^{N}
\left\{
(\mathbf{x}_n - \bar{\mathbf{x}})^\top \mathbf{u}_1
\right\}^2.
$$

Rewrite each squared scalar as a quadratic form:
$$
\left\{
(\mathbf{x}_n - \bar{\mathbf{x}})^\top \mathbf{u}_1
\right\}^2
=
\mathbf{u}_1^\top
(\mathbf{x}_n - \bar{\mathbf{x}})
(\mathbf{x}_n - \bar{\mathbf{x}})^\top
\mathbf{u}_1.
$$

Substituting this into the average gives
$$
\frac{1}{N}
\sum_{n=1}^{N}
\left\{
\mathbf{x}_n^\top \mathbf{u}_1
-
\bar{\mathbf{x}}^\top \mathbf{u}_1
\right\}^2
=
\mathbf{u}_1^\top S\mathbf{u}_1.
\tag{12.2}
$$

The key point is:

> The variance of the data after projection onto $\mathbf{u}_1$ is $\mathbf{u}_1^\top S\mathbf{u}_1$.

Now that the projected variance has been written as a compact matrix expression, the PCA objective is straightforward: choose the unit vector $\mathbf{u}_1$ that makes this quantity as large as possible.

### Maximising Projected Variance

PCA chooses the first principal direction by maximising projected variance:
$$
\max_{\mathbf{u}_1}
\quad
\mathbf{u}_1^\top S\mathbf{u}_1
$$
subject to the unit-length constraint
$$
\mathbf{u}_1^\top \mathbf{u}_1 = 1.
$$

The constraint matters because without it the objective could be made arbitrarily large just by scaling $\mathbf{u}_1$.

Using a Lagrange multiplier, the constrained objective is
$$
\mathbf{u}_1^\top S\mathbf{u}_1
+
\lambda_1(1 - \mathbf{u}_1^\top \mathbf{u}_1).
\tag{12.4}
$$

Differentiate with respect to $\mathbf{u}_1$:
$$
\frac{\partial}{\partial \mathbf{u}_1}
\left[
\mathbf{u}_1^\top S\mathbf{u}_1
+
\lambda_1(1 - \mathbf{u}_1^\top \mathbf{u}_1)
\right]
=
2S\mathbf{u}_1 - 2\lambda_1\mathbf{u}_1.
$$
Setting this equal to zero gives
$$
2S\mathbf{u}_1 - 2\lambda_1\mathbf{u}_1 = 0,
$$
and therefore
$$
S\mathbf{u}_1 = \lambda_1\mathbf{u}_1.
$$

So the first principal component must be an eigenvector of the covariance matrix $S$.

This tells us the possible candidate directions, but not yet which candidate to use first. The final step is to connect the value of the objective back to the corresponding eigenvalue.

### Why Choose the Largest Eigenvalue?

The projected variance is
$$
\mathbf{u}_1^\top S\mathbf{u}_1.
$$
Using the eigenvalue equation,
$$
S\mathbf{u}_1 = \lambda_1\mathbf{u}_1,
$$
we get
$$
\mathbf{u}_1^\top S\mathbf{u}_1
=
\mathbf{u}_1^\top \lambda_1\mathbf{u}_1
=
\lambda_1 \mathbf{u}_1^\top \mathbf{u}_1.
$$
Since $\mathbf{u}_1^\top \mathbf{u}_1 = 1$,
$$
\mathbf{u}_1^\top S\mathbf{u}_1 = \lambda_1.
$$

So the variance captured by direction $\mathbf{u}_1$ is exactly its eigenvalue $\lambda_1$. To maximise projected variance, choose the eigenvector with the largest eigenvalue.

The remaining principal components are the eigenvectors with the next largest eigenvalues, subject to being orthogonal to the components already chosen.

So the maximum-variance derivation has converted the original compression question into a ranking problem over eigenvectors: sort the covariance matrix eigenvectors by eigenvalue, then keep the top $M$ directions.

### Summary

Using the slide notation,
$$
S =
\frac{1}{N}
\sum_{n=1}^{N}
(\mathbf{x}_n - \bar{\mathbf{x}})
(\mathbf{x}_n - \bar{\mathbf{x}})^\top.
$$

The projected variance along $\mathbf{u}_1$ is
$$
\frac{1}{N}
\sum_{n=1}^{N}
\left\{
\mathbf{x}_n^\top \mathbf{u}_1
-
\bar{\mathbf{x}}^\top \mathbf{u}_1
\right\}^2
=
\mathbf{u}_1^\top S\mathbf{u}_1.
$$

PCA solves
$$
\max_{\mathbf{u}_1} \mathbf{u}_1^\top S\mathbf{u}_1
\quad
\text{subject to}
\quad
\mathbf{u}_1^\top \mathbf{u}_1 = 1.
$$

The solution satisfies
$$
S\mathbf{u}_1 = \lambda_1\mathbf{u}_1.
$$

Meaning:

> The first principal component is the eigenvector of $S$ with the largest eigenvalue. That eigenvalue is the variance captured by projecting onto that direction.

## PCA Review: Minimum Error Formulation

The maximum-variance formulation asks which directions preserve the most variance after projection. The minimum-error formulation asks the same question from the reconstruction side:

> Which lower-dimensional reconstruction gives the smallest squared error?

Both views lead to the same PCA solution. Maximum variance says to keep the directions with the largest eigenvalues. Minimum reconstruction error says to discard the directions with the smallest eigenvalues.

### Full Reconstruction

Start again from the full orthonormal basis
$$
\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_D.
$$
Using all $D$ basis directions, any data point can be reconstructed exactly:
$$
\mathbf{x}_n =
\sum_{i=1}^{D}
(\mathbf{x}_n^\top \mathbf{u}_i)\mathbf{u}_i.
\tag{12.9}
$$

There is no information loss here because no directions have been removed. This is just a change of coordinates inside the original $D$-dimensional space.

### Lower-Dimensional Reconstruction

PCA compresses the data by keeping only the first $M$ directions, where
$$
M < D.
$$
The kept directions are
$$
\mathbf{u}_1, \dots, \mathbf{u}_M.
$$

For centred data, the reconstruction can be written simply as
$$
\tilde{\mathbf{x}}_n =
\sum_{i=1}^{M}
(\mathbf{x}_n^\top \mathbf{u}_i)\mathbf{u}_i.
$$

With the Bishop slide convention, however, the sample mean is kept explicitly. This means the discarded coordinates are not set to zero; they are replaced by their mean values:
$$
\bar{\mathbf{x}}^\top \mathbf{u}_i,
\qquad
i = M+1, \dots, D.
$$
So the reconstruction is
$$
\tilde{\mathbf{x}}_n =
\sum_{i=1}^{M}
(\mathbf{x}_n^\top \mathbf{u}_i)\mathbf{u}_i
+
\sum_{i=M+1}^{D}
(\bar{\mathbf{x}}^\top \mathbf{u}_i)\mathbf{u}_i.
$$

The reconstruction error therefore comes only from the directions that were discarded:
$$
\mathbf{x}_n - \tilde{\mathbf{x}}_n
=
\sum_{i=M+1}^{D}
\left\{
\mathbf{x}_n^\top \mathbf{u}_i
-
\bar{\mathbf{x}}^\top \mathbf{u}_i
\right\}
\mathbf{u}_i.
$$

So the core intuition is:

> PCA error is the variation left in the directions we throw away.

### Reconstruction Error Objective

The minimum-error formulation measures average squared reconstruction error:
$$
J =
\frac{1}{N}
\sum_{n=1}^{N}
\|\mathbf{x}_n - \tilde{\mathbf{x}}_n\|^2.
$$

Because the basis vectors are orthonormal, squared error separates cleanly across the discarded directions:
$$
J =
\frac{1}{N}
\sum_{n=1}^{N}
\sum_{i=M+1}^{D}
\left\{
\mathbf{x}_n^\top \mathbf{u}_i
-
\bar{\mathbf{x}}^\top \mathbf{u}_i
\right\}^2.
\tag{12.15}
$$

This is the same projected-variance expression as before, but now summed over the discarded directions. For each discarded direction $\mathbf{u}_i$,
$$
\frac{1}{N}
\sum_{n=1}^{N}
\left\{
\mathbf{x}_n^\top \mathbf{u}_i
-
\bar{\mathbf{x}}^\top \mathbf{u}_i
\right\}^2
=
\mathbf{u}_i^\top S\mathbf{u}_i.
$$

Therefore the reconstruction error can be written compactly as
$$
J =
\sum_{i=M+1}^{D}
\mathbf{u}_i^\top S\mathbf{u}_i.
\tag{12.15}
$$

This says that the total reconstruction error is the sum of the variances in the directions we did not keep.

### Error as Sum of Discarded Eigenvalues

Using the eigenvalue equation
$$
S\mathbf{u}_i = \lambda_i \mathbf{u}_i,
$$
each term becomes
$$
\mathbf{u}_i^\top S\mathbf{u}_i
=
\mathbf{u}_i^\top \lambda_i \mathbf{u}_i
=
\lambda_i \mathbf{u}_i^\top \mathbf{u}_i.
$$
Since each eigenvector has unit length,
$$
\mathbf{u}_i^\top \mathbf{u}_i = 1,
$$
we get
$$
\mathbf{u}_i^\top S\mathbf{u}_i = \lambda_i.
$$

So the reconstruction error is
$$
J =
\sum_{i=M+1}^{D}
\lambda_i.
\tag{12.18}
$$

Equivalently,
$$
J =
\lambda_{M+1}
+
\lambda_{M+2}
+
\dots
+
\lambda_D.
$$

The error is the sum of the eigenvalues corresponding to the discarded directions.

### Meaning

To minimise reconstruction error, PCA should discard the directions with the smallest eigenvalues. Equivalently, it should keep the directions with the largest eigenvalues:
$$
\mathbf{u}_1, \dots, \mathbf{u}_M.
$$

This is the same decision as the maximum-variance formulation:

- Maximum variance projection says: keep the directions with largest variance.
- Minimum error formulation says: discard the directions with smallest variance.

These are two sides of the same argument.

The intuition is easiest to see with a long, thin data cloud. If we keep the long-axis direction, a one-dimensional reconstruction is good because most of the variation lies along that axis. If we keep the short-axis direction instead, reconstruction is poor because the long-axis variation is thrown away. Reconstruction error is small precisely when the kept directions contain most of the data variation.

### Connection to Maximum Variance

The total variance in the data is fixed:
$$
\sum_{i=1}^{D}
\lambda_i.
$$

Maximum variance projection tries to maximise the kept variance:
$$
\sum_{i=1}^{M}
\lambda_i.
$$

Minimum reconstruction error tries to minimise the lost variance:
$$
\sum_{i=M+1}^{D}
\lambda_i.
$$

Since
$$
\sum_{i=1}^{D}
\lambda_i
=
\sum_{i=1}^{M}
\lambda_i
+
\sum_{i=M+1}^{D}
\lambda_i,
$$
maximising the kept variance is exactly the same as minimising the discarded variance.

## Probabilistic PCA: Likelihood and Posterior

PPCA turns PCA into a Gaussian latent-variable model. Instead of only defining a projection geometrically, we assume each observed data point $\mathbf{x}$ is generated from a lower-dimensional latent variable $\mathbf{z}$.

The model is
$$
p(\mathbf{z}) = \mathcal{N}(\mathbf{z}\mid \mathbf{0}, I),
$$
and
$$
p(\mathbf{x}\mid \mathbf{z})
=
\mathcal{N}(\mathbf{x}\mid W\mathbf{z}+\boldsymbol{\mu}, \sigma^2 I).
$$

Equivalently,
$$
\mathbf{x} = W\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim \mathcal{N}(\mathbf{0}, \sigma^2 I).
$$

Here
$$
\mathbf{z}\in\mathbb{R}^M,
\qquad
\mathbf{x}\in\mathbb{R}^D,
\qquad
W\in\mathbb{R}^{D\times M},
\qquad
M<D.
$$

The generative story is:

> sample a low-dimensional latent point $\mathbf{z}$, map it into data space using $W$, shift by $\boldsymbol{\mu}$, then add spherical Gaussian noise.

### Marginal Likelihood of \(\mathbf{x}\)

To get the distribution of the observed data point, integrate out the latent variable:
$$
p(\mathbf{x})
=
\int p(\mathbf{x}\mid \mathbf{z})p(\mathbf{z})\,d\mathbf{z}.
$$

Because the prior and conditional likelihood are Gaussian, the marginal distribution is also Gaussian:
$$
p(\mathbf{x})
=
\mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu}, C),
\tag{12.35}
$$
where
$$
C = WW^\top + \sigma^2 I.
\tag{12.36}
$$

This covariance has two parts:

- $WW^\top$ is structured covariance caused by low-dimensional latent variation.
- $\sigma^2 I$ is isotropic noise around the latent subspace.

So PPCA says that most covariance in $\mathbf{x}$ comes from a low-dimensional latent source, with spherical Gaussian noise added around it.

### Why \(C = WW^\top + \sigma^2 I\)?

From the generative equation,
$$
\mathbf{x} = W\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}.
$$

The mean is
$$
\mathbb{E}[\mathbf{x}]
=
\mathbb{E}[W\mathbf{z}+\boldsymbol{\mu}+\boldsymbol{\epsilon}]
=
\boldsymbol{\mu},
\tag{12.37}
$$
because
$$
\mathbb{E}[\mathbf{z}] = \mathbf{0},
\qquad
\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}.
$$

The covariance is
$$
\operatorname{cov}[\mathbf{x}]
=
\mathbb{E}[(W\mathbf{z}+\boldsymbol{\epsilon})(W\mathbf{z}+\boldsymbol{\epsilon})^\top].
$$

The cross terms vanish because $\mathbf{z}$ and $\boldsymbol{\epsilon}$ are independent and zero-mean, so
$$
\operatorname{cov}[\mathbf{x}]
=
\mathbb{E}[W\mathbf{z}\mathbf{z}^\top W^\top]
+
\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top].
$$

Using
$$
\mathbb{E}[\mathbf{z}\mathbf{z}^\top] = I,
\qquad
\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top] = \sigma^2 I,
$$
we obtain
$$
\operatorname{cov}[\mathbf{x}]
=
WW^\top + \sigma^2 I.
\tag{12.38}
$$

### Posterior Distribution of \(\mathbf{z}\)

After observing $\mathbf{x}$, we can infer a distribution over its latent coordinate $\mathbf{z}$:
$$
p(\mathbf{z}\mid \mathbf{x}).
$$

The posterior is Gaussian:
$$
p(\mathbf{z}\mid \mathbf{x})
=
\mathcal{N}
\left(
\mathbf{z}
\mid
\mathbf{M}^{-1}W^\top(\mathbf{x}-\boldsymbol{\mu}),
\sigma^2 \mathbf{M}^{-1}
\right),
\tag{12.42}
$$
where
$$
\mathbf{M} = W^\top W + \sigma^2 I.
$$

The most important part is the posterior mean:
$$
\mathbb{E}[\mathbf{z}\mid \mathbf{x}]
=
\mathbf{M}^{-1}W^\top(\mathbf{x}-\boldsymbol{\mu}).
$$

This is the probabilistic version of the PCA projection. Ordinary PCA gives a single code such as
$$
\mathbf{z} = U^\top \mathbf{x},
$$
whereas PPCA gives a full posterior distribution over possible latent codes.

### Why Woodbury Appears

The marginal covariance is
$$
C = WW^\top + \sigma^2 I,
$$
which is a $D\times D$ matrix. Directly inverting $C$ can be expensive when $D$ is large.

The Woodbury identity lets us write
$$
C^{-1}
=
\sigma^{-2}I
-
\sigma^{-2}W \mathbf{M}^{-1}W^\top,
$$
where
$$
\mathbf{M} = W^\top W+\sigma^2I.
$$

The useful point is computational: instead of inverting a large $D\times D$ matrix, we only invert the smaller $M\times M$ matrix $\mathbf{M}$. This matters because PPCA usually assumes $M\ll D$.

## Probabilistic PCA: Maximum Likelihood

Once the latent variable has been integrated out, the observed data distribution is
$$
p(\mathbf{x})
=
\mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu}, C),
\qquad
C = WW^\top+\sigma^2I.
$$

For a dataset
$$
X = \{\mathbf{x}_1,\dots,\mathbf{x}_N\},
$$
assuming i.i.d. observations, the log likelihood is
$$
\ln p(X\mid \boldsymbol{\mu}, W, \sigma^2)
=
\sum_{n=1}^{N}
\ln p(\mathbf{x}_n\mid \boldsymbol{\mu}, W, \sigma^2).
$$

Since each observation is Gaussian, this becomes
$$
\ln p(X\mid \boldsymbol{\mu}, W, \sigma^2)
=
-
\frac{ND}{2}\ln(2\pi)
-
\frac{N}{2}\ln|C|
-
\frac{1}{2}
\sum_{n=1}^{N}
(\mathbf{x}_n-\boldsymbol{\mu})^\top C^{-1}(\mathbf{x}_n-\boldsymbol{\mu}).
\tag{12.43}
$$

The parameters are
$$
\boldsymbol{\mu},
\qquad
W,
\qquad
\sigma^2.
$$
Maximising the likelihood means choosing the mean, latent loading matrix, and isotropic noise variance so that the Gaussian covariance
$$
C = WW^\top+\sigma^2I
$$
matches the observed covariance structure of the data.

### Maximum Likelihood Estimate of $\boldsymbol{\mu}$

The maximum likelihood estimate of the mean is the sample mean:
$$
\boldsymbol{\mu}_{\mathrm{ML}}
=
\bar{\mathbf{x}}
=
\frac{1}{N}
\sum_{n=1}^{N}
\mathbf{x}_n.
$$

This matches ordinary PCA: first centre the data.

### Maximum Likelihood Estimate of \(W\)

The maximum likelihood solution for $W$ is
$$
W_{\mathrm{ML}}
=
U_M(L_M-\sigma^2I)^{1/2}R.
\tag{12.45}
$$

Here
$$
U_M = [\mathbf{u}_1,\dots,\mathbf{u}_M]
$$
contains the top $M$ eigenvectors of the sample covariance matrix, and
$$
L_M =
\operatorname{diag}(\lambda_1,\dots,\lambda_M)
$$
contains the corresponding top $M$ eigenvalues. The matrix $R$ is an arbitrary $M\times M$ orthogonal rotation matrix.

The key meaning is:

> $W_{\mathrm{ML}}$ spans the same subspace as the top $M$ PCA eigenvectors.

So PPCA recovers the same principal subspace as ordinary PCA. The factor
$$
(L_M-\sigma^2I)^{1/2}
$$
scales each principal direction according to how much variance remains after subtracting isotropic noise.

Ordinary PCA gives the principal basis directions $U_M$. PPCA gives a loading matrix $W_{\mathrm{ML}}$ whose column space is the same principal subspace, but scaled and possibly rotated within that subspace.

### Why the Rotation Matrix \(R\) Appears

The latent prior
$$
p(\mathbf{z})=\mathcal{N}(\mathbf{z}\mid \mathbf{0},I)
$$
is rotationally symmetric. Rotating the latent coordinate system does not change the model distribution.

If $R$ is orthogonal, then replacing $W$ by $WR$ gives
$$
(WR)(WR)^\top
=
WRR^\top W^\top
=
WW^\top,
$$
because
$$
RR^\top = I.
$$

Therefore $p(\mathbf{x})$ does not change. The model identifies the principal subspace, not a unique orientation of coordinates inside the latent space.

### Maximum Likelihood Estimate of $\sigma^2$

The maximum likelihood estimate of the noise variance is
$$
\sigma^2_{\mathrm{ML}}
=
\frac{1}{D-M}
\sum_{i=M+1}^{D}
\lambda_i.
\tag{12.46}
$$

This is the average of the discarded eigenvalues.

In ordinary PCA, discarded variance is treated as reconstruction error:
$$
J =
\sum_{i=M+1}^{D}
\lambda_i.
$$

In PPCA, the discarded variance becomes isotropic Gaussian noise:
$$
\sigma^2I.
$$

That is the bridge between PCA and PPCA: PCA loses the small-eigenvalue directions as reconstruction error, while PPCA models their average effect as spherical noise around the principal subspace.

### Summary

The flow from likelihood to maximum likelihood is:
$$
\mathbf{z}\sim \mathcal{N}(\mathbf{0},I),
$$
$$
\mathbf{x}\mid \mathbf{z}
\sim
\mathcal{N}(W\mathbf{z}+\boldsymbol{\mu},\sigma^2I).
$$

Integrating out $\mathbf{z}$ gives
$$
p(\mathbf{x})
=
\mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu}, WW^\top+\sigma^2I).
$$

Maximising the likelihood gives
$$
\boldsymbol{\mu}_{\mathrm{ML}} = \bar{\mathbf{x}},
$$
$$
W_{\mathrm{ML}}
=
U_M(L_M-\sigma^2I)^{1/2}R,
$$
and
$$
\sigma^2_{\mathrm{ML}}
=
\frac{1}{D-M}
\sum_{i=M+1}^{D}
\lambda_i.
$$

Exam-level takeaway:

> PPCA is a Gaussian latent-variable model whose maximum likelihood solution recovers the same principal subspace as PCA. The top eigenvectors/eigenvalues explain structured variation, while the discarded eigenvalues estimate the isotropic noise variance.

## Factor Analysis Connection

PPCA and Factor Analysis are both linear-Gaussian latent variable models. In both cases, a low-dimensional latent vector explains the main structured variation in the observed high-dimensional data.

The key difference is the noise covariance.

PPCA uses spherical noise:
$$
\sigma^2 I.
$$
This means every observed dimension has the same noise variance.

Factor Analysis uses diagonal noise:
$$
\Psi,
$$
where
$$
\Psi =
\begin{bmatrix}
\psi_1 & 0 & \cdots & 0 \\
0 & \psi_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \psi_D
\end{bmatrix}.
$$

So Factor Analysis allows each observed dimension to have its own noise variance:
$$
\psi_1,\psi_2,\dots,\psi_D.
$$

The comparison is:

- PPCA: spherical noise, $\sigma^2 I$.
- Factor Analysis: diagonal noise, $\Psi$.

So Factor Analysis is more flexible in how it models observation noise, while PPCA is the more restricted model. The shared idea is still the same: explain high-dimensional observations using a lower-dimensional latent factor model plus Gaussian noise.
