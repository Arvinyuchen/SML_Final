In week2&3, we considered linear parametric models for regression and classification in which the form of the mapping $y(\mathbf{x}, \mathbf{w})$ from input $\mathbf{x}$ to output $y$ is governed by a vector $\mathbf{w}$ of adaptive parameters. During the learning phase, a set of training data is used either to obtain a point estimate of the parameter vector or to determine a posterior distribution over this vector. The training data is then **discarded**, and predictions for new inputs are based purely on the learned parameter vector $\mathbf{w}$. 

However, there is a class of pattern recognition techniques, in which the training data points, or subset of them, are kept and used also during the prediction phase. Many linear parametric models can be **re-cast** into an equivalent "dual representation" in which the predictions are also based on linear combinations of a *kernel function* evaluated at the training data points.

## Prerequisite

### Jacobian matrix (vector-to-vector derivative)
Given an input vector $\mathbf{x} \in \mathbb{R}^n$ (as a column vector) and a function $\mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$ , the derivative of $\mathbf{f}$  with respective to $\mathbf{x}$ is a $m\times n$ matrix, also known as Jacobian matrix:
$$
\mathbf{J}
=
\frac{\partial \mathbf{f}}{\partial \mathbf{x}}
=
\begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n}
\\[0.8em]
\vdots & \ddots & \vdots
\\[0.8em]
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix}
\in \mathbb{R}^{m\times n}
$$
The gradient of a vector with respect to a vector is defined as $\nabla_{\mathbf{x}} \mathbf{f} = \mathbf{J}^T \in \mathbb{R}^{n\times m}$ and this formation is also valid when $m=1$ (i.e., scalar output).

## Dual Representations
Many linear models for regression and classification can be reformulated in terms of a dual representation in which the kernel function arises naturally. This concept will play an important role when we consider support vector machines. 

Here we consider a linear regression model whose parameters are determined by minimising a regularised sum-of-squared error function given by
$$
J(\mathbf{w})
=
\frac{1}{2}
\sum_{n=1}^{N}
\left\{
\mathbf{w}^{\mathrm T}\boldsymbol{\phi}(\mathbf{x}_n)-t_n
\right\}^{2}
+
\frac{\lambda}{2}
\mathbf{w}^{\mathrm T}\mathbf{w}
\tag{6.2}
$$
where $\lambda \geq 0.$ If we set the gradient of $J(\mathbf{w})$ with respect to $\mathbf{w}$ equal to zero, we see that the solution for $\mathbf{w}$ takes the form of a linear combination of the vectors $\phi(\mathbf{x}_n)$, with coefficients that are functions of $\mathbf{w}$, of the form
$$
\mathbf{w}
=
-\frac{1}{\lambda}
\sum_{n=1}^{N}
\left\{
\mathbf{w}^{\mathrm T}\boldsymbol{\phi}(\mathbf{x}_n)-t_n
\right\}
\boldsymbol{\phi}(\mathbf{x}_n)
=
\sum_{n=1}^{N}
a_n \boldsymbol{\phi}(\mathbf{x}_n)
=
\mathbf{\Phi}^{\mathrm T}\mathbf{a}
\tag{6.3}
$$where $\mathbf{\Phi}$ is the design matrix, whose $n^{\text{th}}$ row is given by $\boldsymbol{\phi}(\mathbf{x}_n)^T$. here the vector $\mathbf{a} = (a_1, \dots, a_N)^T$, we have defined

$$
a_n =
-\frac{1}{\lambda}
\left\{
\mathbf{w}^{\mathrm T}\boldsymbol{\phi}(\mathbf{x}_n)-t_n
\right\}.
\tag{6.4}
$$
Instead of working with the parameter vector $\mathbf{w}$, we can now reformulate the least-squares algorithm in terms of the parameter vector $\mathbf{a}$, giving rise to a *dual representation*. If we substitute $\mathbf{w} = \mathbf{\Phi}^T\mathbf{a}$ into $J(\mathbf{w})$, we obtain

$$
J(\mathbf{a})=
\frac{1}{2}
\mathbf{a}^{\mathrm T}
\mathbf{\Phi}
\mathbf{\Phi}^{\mathrm T}
\mathbf{\Phi}
\mathbf{\Phi}^{\mathrm T}
\mathbf{a}
-
\mathbf{a}^{\mathrm T}
\mathbf{\Phi}
\mathbf{\Phi}^{\mathrm T}
\mathbf{t}
+
\frac{1}{2}
\mathbf{t}^{\mathrm T}\mathbf{t}
+
\frac{\lambda}{2}
\mathbf{a}^{\mathrm T}
\mathbf{\Phi}
\mathbf{\Phi}^{\mathrm T}
\mathbf{a}
\tag{6.5}
$$
where $\mathbf{t} = (t_1, \dots, t_N)^T$.  We now define the *Gram matrix* $\mathbf{K} = \mathbf{\Phi}^{\mathrm T}\mathbf{\Phi}$, which is an $N \times N$ symmetric matrix with element s
$$
K_{nm}
=
\boldsymbol{\phi}(\mathbf{x}_n)^{\mathrm T}
\boldsymbol{\phi}(\mathbf{x}_m)
=
k(\mathbf{x}_n,\mathbf{x}_m)
\tag{6.6}
$$
which is defined as the *kernel function*. In terms of the Gram matrix, the sum-of-squares error function can be written as
$$
J(\mathbf{a})
=
\frac{1}{2}
\mathbf{a}^{\mathrm T}
\mathbf{K}
\mathbf{K}
\mathbf{a}
-
\mathbf{a}^{\mathrm T}
\mathbf{K}
\mathbf{t}
+
\frac{1}{2}
\mathbf{t}^{\mathrm T}\mathbf{t}
+
\frac{\lambda}{2}
\mathbf{a}^{\mathrm T}
\mathbf{K}
\mathbf{a}.
\tag{6.7}
$$
Setting the gradient of $J(\mathbf{a})$ with respect to $\mathbf{a}$ to zero, we obtain the following solution
$$
\mathbf{a}
=
\left(
\mathbf{K}
+
\lambda \mathbf{I}_N
\right)^{-1}
\mathbf{t}.
\tag{6.8}
$$
If we substitute this back into the linear regression model, we obtain the following prediction for a new input $\mathbf{x}$
$$
y(\mathbf{x})
=
\mathbf{w}^{\mathrm T}\boldsymbol{\phi}(\mathbf{x})
=
\mathbf{a}^{\mathrm T}\mathbf{\Phi}\,\boldsymbol{\phi}(\mathbf{x})
=
\mathbf{k}(\mathbf{x})^{\mathrm T}
\left(
\mathbf{K}
+
\lambda \mathbf{I}_N
\right)^{-1}
\mathbf{t}
\tag{6.9}
$$
where we have defined the vector $\mathbf{k}(\mathbf{x})$ with elements $k_n(\mathbf{x}) = k(\mathbf{x}_n, \mathbf{x})$.  Thus we see that the dual formulation allows the solution to the least-squares problem to be expressed entirely in terms of the kernel function $k(\mathbf{x},\mathbf{x}')$. 

In the lecture slides, a algebraic-identity method is used to derive the same result as (6.9).
$$
\begin{aligned}
y(\mathbf{x})
&=
\boldsymbol{\phi}(\mathbf{x})^{\mathrm T}\mathbf{w}^{*}
\\
&=
\boldsymbol{\phi}(\mathbf{x})^{\mathrm T}
\left(
\lambda \mathbf{I}
+
\mathbf{\Phi}^{\mathrm T}\mathbf{\Phi}
\right)^{-1}
\mathbf{\Phi}^{\mathrm T}\mathbf{t}
\\
&=
\boldsymbol{\phi}(\mathbf{x})^{\mathrm T}
\left(
\lambda \mathbf{I}
+
\mathbf{\Phi}^{\mathrm T}\mathbf{\Phi}
\right)^{-1}
\mathbf{\Phi}^{\mathrm T}
\left(
\lambda \mathbf{I}
+
\mathbf{\Phi}\mathbf{\Phi}^{\mathrm T}
\right)
\left(
\lambda \mathbf{I}
+
\mathbf{\Phi}\mathbf{\Phi}^{\mathrm T}
\right)^{-1}
\mathbf{t}
\\
&=
\boldsymbol{\phi}(\mathbf{x})^{\mathrm T}
\left(
\lambda \mathbf{I}
+
\mathbf{\Phi}^{\mathrm T}\mathbf{\Phi}
\right)^{-1}
\left(
\lambda \mathbf{\Phi}^{\mathrm T}
+
\mathbf{\Phi}^{\mathrm T}\mathbf{\Phi}\mathbf{\Phi}^{\mathrm T}
\right)
\left(
\lambda \mathbf{I}
+
\mathbf{\Phi}\mathbf{\Phi}^{\mathrm T}
\right)^{-1}
\mathbf{t}
\\
&=
\boldsymbol{\phi}(\mathbf{x})^{\mathrm T}
\left(
\lambda \mathbf{I}
+
\mathbf{\Phi}^{\mathrm T}\mathbf{\Phi}
\right)^{-1}
\left(
\lambda \mathbf{I}
+
\mathbf{\Phi}^{\mathrm T}\mathbf{\Phi}
\right)
\mathbf{\Phi}^{\mathrm T}
\left(
\lambda \mathbf{I}
+
\mathbf{\Phi}\mathbf{\Phi}^{\mathrm T}
\right)^{-1}
\mathbf{t}
\\
&=
\boldsymbol{\phi}(\mathbf{x})^{\mathrm T}
\mathbf{\Phi}^{\mathrm T}
\left(
\lambda \mathbf{I}
+
\mathbf{\Phi}\mathbf{\Phi}^{\mathrm T}
\right)^{-1}
\mathbf{t}
\\
&=
\mathbf{k}(\mathbf{x})^{\mathrm T}
\left(
\lambda \mathbf{I}
+
\mathbf{K}
\right)^{-1}
\mathbf{t}.
\end{aligned}
$$

The advantage of the dual formulation, as we shall see, is that it is expressed entirely in terms of the kernel function $k(\mathbf{x}, \mathbf{x}')$. We can therefore work directly in terms of kernels and avoid the explicit introduction of the feature vector $\boldsymbol{\phi}(\mathbf{x})$, which allows us implicitly to use feature spaces of high , even infinite, dimensionality.

## Constructing Kernels
In order to exploit kernel substitution, we need to be able to construct valid kernel functions. One approach is to choose a feature space mapping $\boldsymbol{\phi}(\mathbf{x})$ and then use this to find the corresponding kernel, as illustrated in Figure 6.1
![[Pasted image 20260529120542.png]]
Here the kernel function is defined for a one-dimensional input space by
$$
k(x,x')= \boldsymbol{\phi}(x)^T\boldsymbol{\phi}(x') = \sum^M_{i=1} \phi_i(x)\phi_i(x')
$$
where $\phi_i(x)$ are the basis functions. 

Alternatively, we can construct kernel functions directly. In this case, we must ensure that the function we choose is a valid kernel, in other words that it corresponds to a scalar product in some feature space. As a simple example, consider a kernel function given by
$$
k(\mathbf{x},\mathbf{z}) = (\mathbf{x}^T\mathbf{z})^2.
$$

If we take the particular case of a two-dimensional input space $\mathbf{x}=(x_1, x_2)$ we can expand out the terms and thereby identify the corresponding nonlinear feature mapping
$$
\begin{aligned}
k(\mathbf{x},\mathbf{z})
&=
(\mathbf{x}^{\mathrm T}\mathbf{z})^2
=
(x_1z_1+x_2z_2)^2
\\
&=
x_1^2z_1^2
+
2x_1z_1x_2z_2
+
x_2^2z_2^2
\\
&=
(x_1^2,\sqrt{2}x_1x_2,x_2^2)
(z_1^2,\sqrt{2}z_1z_2,z_2^2)^{\mathrm T}
\\
&=
\boldsymbol{\phi}(\mathbf{x})^{\mathrm T}
\boldsymbol{\phi}(\mathbf{z}).
\end{aligned}
\tag{6.12}
$$
We see that the feature mapping takes the form $\boldsymbol{\phi}(\mathbf{x}) = (x_1^2, \sqrt{2}x_1x_2, x_2^2)^T$ and therefore comprises all possible second order terms, with a specific weighting between them.

A necessary and sufficient condition for a function $k(\mathbf{x}, \mathbf{x}')$ to be a valid kernel is that the Gram matrix $\mathbf{K}$, whose elements are given by $k(\mathbf{x}_n, \mathbf{x}_m)$, should be positive semidefinite for all possible choices of the set $\{\mathbf{x}_n\}$. 

Any Gram matrix of an inner-product feature map is PSD. With $k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}')$ and $K_{ij} = \boldsymbol{\phi}(\mathbf{x}_i)^T \boldsymbol{\phi}(\mathbf{x}_j)$, take any $\mathbf{c} \in \mathbb{R}^n$:
$$
\mathbf{c}^{\mathrm T}\mathbf{K}\mathbf{c}
=
\sum_{i=1}^{n}\sum_{j=1}^{n}
c_i c_j\,
\boldsymbol{\phi}(\mathbf{x}_i)^{\mathrm T}
\boldsymbol{\phi}(\mathbf{x}_j)
=
\left(
\sum_{i=1}^{n}
c_i\boldsymbol{\phi}(\mathbf{x}_i)
\right)^{\mathrm T}
\left(
\sum_{j=1}^{n}
c_j\boldsymbol{\phi}(\mathbf{x}_j)
\right)
=
\left\|
\sum_{i=1}^{n}
c_i\boldsymbol{\phi}(\mathbf{x}_i)
\right\|^2
\ge 0.
$$
However, the kernel tricks's whole point is to skip $\boldsymbol{\phi}$ and write $k$ down directly, as a function of $\mathbf{x}$ and $\mathbf{x}'$ without knowing whether any feature map produces it. Therefore one powerful techniqiue for constructing new kernels is to build them out of simpler kernels as building blocks. This can be done using the following properties:
![[Pasted image 20260529123831.png|541]]

## Common kernels
The Gaussian / RBF kernel:
$$
k(\mathbf{x}, \mathbf{x}') = \exp \left(- \frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\sigma^2} \right) \tag{6.23}.
$$
We can see that this is a valid kernel by expanding the square
$$
\| \mathbf{x} - \mathbf{x}' \| = \mathbf{x}^T \mathbf{x} + (\mathbf{x}')^T \mathbf{x}' - 2 \mathbf{x}^T\mathbf{x} \tag{6.24}
$$
to give 
$$
k(\mathbf{x},\mathbf{x}')
=
\exp\!\left(
-\frac{\mathbf{x}^{\mathrm T}\mathbf{x}}{2\sigma^2}
\right)
\exp\!\left(
\frac{\mathbf{x}^{\mathrm T}\mathbf{x}'}{\sigma^2}
\right)
\exp\!\left(
-\frac{(\mathbf{x}')^{\mathrm T}\mathbf{x}'}{2\sigma^2}
\right)
\tag{6.25}
$$
and then making use of (6.14) and (6.16), together with the validity of the linear kernel $k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T\mathbf{x}'$. Note that the feature vector that corresponds to the Gaussian kernel has infinite dimensionality. 
$$
\boldsymbol\phi(x) = \exp\!\left(-\frac{x^2}{2\sigma^2}\right)\left[1,\; \frac{x}{\sigma\sqrt{1!}},\; \frac{x^2}{\sigma^2\sqrt{2!}},\; \frac{x^3}{\sigma^3\sqrt{3!}},\; \dots\right]^\top.
$$
**More examples of kernels**
$$
\begin{aligned}
k(\mathbf{x},\mathbf{x}')
&=
(\mathbf{x}^{\mathrm T}\mathbf{x}')^M
&& \text{only terms of degree } M
\\[0.5em]
k(\mathbf{x},\mathbf{x}')
&=
(\mathbf{x}^{\mathrm T}\mathbf{x}' + c)^M
&& \text{all terms up to degree } M
\\[0.5em]
k(\mathbf{x},\mathbf{x}')
&=
\exp\!\left(
-\frac{\|\mathbf{x}-\mathbf{x}'\|^2}{2\sigma^2}
\right)
&& \text{Gaussian kernel}
\\[0.5em]
k(\mathbf{x},\mathbf{x}')
&=
\tanh\!\left(
a\,\mathbf{x}^{\mathrm T}\mathbf{x}' + b
\right)
&& \text{Sigmoidal kernel (invalid)}
\end{aligned}
$$
Generally, we call
$$
\begin{aligned}
k(\mathbf{x},\mathbf{x}')
&=
\mathbf{x}^{\mathrm T}\mathbf{x}'
&& \text{linear kernel}
\\[0.5em]
k(\mathbf{x},\mathbf{x}')
&=
k(\mathbf{x}-\mathbf{x}')
&& \text{stationary kernel}
\\[0.5em]
k(\mathbf{x},\mathbf{x}')
&=
k\!\left(\|\mathbf{x}-\mathbf{x}'\|\right)
&& \text{homogeneous kernel}
\end{aligned}
$$
## Kernel Versatility
Kernels need only a valid similarity measure, so the input need not to be vectors at all.

**Kernels over sets/strings/graphs**
E.g., given a set $\mathcal{A}$ and the set of all subsets of $\mathcal{A}$, called the power set $\mathcal{P}(\mathcal{A})$. For two subsets $\mathcal{A}_1, \mathcal{A}_2 \in \mathcal{P}(\mathcal{A})$, denote the number of elements of the intersection of $\mathcal{A}_1$ and $\mathcal{A}_2$ by $|\mathcal{A}_1 \cap \mathcal{A}_2|$. Then it can be shown that
$$
k(\mathcal{A}_1,\mathcal{A}_2)
=
2^{|\mathcal{A}_1 \cap \mathcal{A}_2|}
$$
corresponds to an inner product in a feature space. Therefore, $k(\mathcal{A}_1,\mathcal{A}_2)$ is a valid kernel function.

**Kernels from probabilistic generative models**
Given $p(\mathbf{x})$, we can define a kernel
$$
k(\mathbf{x},\mathbf{x}')
=
p(\mathbf{x})\,p(\mathbf{x}')
$$
which means two inputs $\mathbf{x}$ and $\mathbf{x}'$ are similar if they both have high probabilities. Include a weighting function $p(i)$ and extend the kernel to
$$
k(\mathbf{x},\mathbf{x}')
=
\sum_i
p(\mathbf{x}\mid i)\,
p(\mathbf{x}'\mid i)\,
p(i).
$$
For a continuous variable $\mathbf{z}$
$$
k(\mathbf{x},\mathbf{x}')
=
\int
p(\mathbf{x}\mid \mathbf{z})\,
p(\mathbf{x}'\mid \mathbf{z})\,
p(\mathbf{z})
\,d\mathbf{z}.
$$
Hidden Markov Model with sequences of length $L$.


## Sparse Kernel Machines

Kernel algorithms have a serious limitation: the kernel $k(\mathbf{x}_n, \mathbf{x}_m)$ must be evaluated for **all pairs** of training points, making the Gram matrix $\mathbf{K}$ an $N \times N$ dense matrix. This is computationally infeasible at training time and slow at prediction time. Recall from (6.9) that kernel ridge regression's prediction
$$
y(\mathbf{x}) = \mathbf{k}(\mathbf{x})^\top (\mathbf{K} + \lambda \mathbf{I}_N)^{-1} \mathbf{t}
$$
sums over *every* training point. We now look at kernel-based algorithms that have **sparse** solutions, so that predictions for new inputs depend only on the kernel function evaluated at a *subset* of the training data points. That subset is the *support vectors*.

We begin by looking in some detail at the *support vector machine* (SVM), a **non-probabilistic, non-parametric** classifier related to kernel logistic regression. The SVM is a decision machine and does not provide posterior probabilities. An important property of SVMs is that the determination of the model parameters corresponds to a **convex optimisation** problem, and so any local solution is also a global optimum — no local-minima worries, unlike neural nets.

### The Linear Model for Two Classes (Bishop 7.1)

The classifier is
$$
y(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}) + b, \tag{7.1}
$$
with feature map $\boldsymbol{\phi}$ (so kernelisable) and **explicit bias** $b$. Bishop makes $b$ explicit deliberately, because the dual will eliminate working in feature space. Two SVM-specific conventions, both different from earlier chapters:
- Targets $t_n \in \{-1, +1\}$, not $\{0,1\}$.
- New points classified by $\text{sign}(y(\mathbf{x}))$.

**The separability assumption.** Assume the data is linearly separable in feature space: there exists at least one choice of parameters $\mathbf{w}$ and $b$ such that $y(\mathbf{x}_n) > 0$ for points with $t_n = +1$ and $y(\mathbf{x}_n) < 0$ for points with $t_n = -1$, so that $t_n y(\mathbf{x}_n) > 0$ for all training points. That single product condition $t_n y(\mathbf{x}_n) > 0$ is the compact statement of "correctly classified," and it is clean precisely *because* of the $\pm 1$ convention.

### The Multiple-Separator Problem (Bishop p. 326)

If the data is separable, infinitely many hyperplanes work. The perceptron finds one, but the solution it finds depends on the arbitrary initial values of $\mathbf{w}$ and $b$ and on the order in which data points are presented. So: of all separating hyperplanes, which generalises best?

The SVM approaches this through the concept of the **margin**, defined as the smallest distance between the decision boundary and any of the samples; the decision boundary is chosen to be the one for which the margin is maximised.


### Constructing the Max-Margin Classifier — the Primal QP

**Critical distinction.** The boundary $y(\mathbf{x})=0$ is a *hyperplane in feature space* $\boldsymbol{\phi}$, but back in the *original input space* $\mathbf{x}$ it is the curve $\{\mathbf{x}: \mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x})+b=0\}$, which is **nonlinear** whenever $\boldsymbol{\phi}$ is nonlinear. The kernel/feature map controls the *shape* of the input-space boundary (linear kernel $\to$ straight; polynomial $\to$ polynomial curve; RBF $\to$ arbitrarily wiggly). "Linear in $\boldsymbol{\phi}$, nonlinear in $\mathbf{x}$" is the entire point of kernelisation.
![[Pasted image 20260530204126.png]]

#### Step 1: Signed distance from a point to the boundary
![[Pasted image 20260530204232.png]]
We need the perpendicular distance from any point $\mathbf{x}$ to the surface $y(\mathbf{x})=0$. Two geometric facts:

**(a) $\mathbf{w}$ is normal to the boundary.** Take any two points $\mathbf{x}_A, \mathbf{x}_B$ on the surface: $y(\mathbf{x}_A)=y(\mathbf{x}_B)=0$. Subtracting gives $\mathbf{w}^\top(\mathbf{x}_A - \mathbf{x}_B)=0$ (the bias cancels). Since $\mathbf{x}_A - \mathbf{x}_B$ is an arbitrary direction *within* the surface, $\mathbf{w}$ is orthogonal to every in-surface direction — i.e. $\mathbf{w}$ is the normal.

**(b) Decomposition.** The normal direction and the surface together span the whole space, so every $\mathbf{x}$ splits uniquely into its orthogonal projection $\mathbf{x}_\perp$ onto the surface (satisfying $y(\mathbf{x}_\perp)=0$) plus a displacement $r$ along the unit normal:
$$
\mathbf{x} = \mathbf{x}_\perp + r\,\frac{\mathbf{w}}{\|\mathbf{w}\|}.
$$
The scalar $r$ is the **signed** distance — positive on the side $\mathbf{w}$ points toward, negative on the other. Substituting into $y$:
$$
y(\mathbf{x}) = \mathbf{w}^\top\!\Big(\mathbf{x}_\perp + r\,\frac{\mathbf{w}}{\|\mathbf{w}\|}\Big) + b = \underbrace{\big(\mathbf{w}^\top\mathbf{x}_\perp + b\big)}_{=\,0\;\text{(since }\mathbf{x}_\perp\text{ is on the surface)}} + r\,\frac{\|\mathbf{w}\|^2}{\|\mathbf{w}\|} = r\|\mathbf{w}\|.
$$
Therefore
$$
r = \frac{y(\mathbf{x})}{\|\mathbf{w}\|}.
$$
For correctly classified points, $\text{sign}(y(\mathbf{x}_n)) = t_n$, so multiplying by $t_n$ turns the signed distance into the positive unsigned distance (Bishop 7.2):
$$
\text{distance of } \mathbf{x}_n \text{ to boundary} = \frac{t_n\, y(\mathbf{x}_n)}{\|\mathbf{w}\|} = \frac{t_n(\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)+b)}{\|\mathbf{w}\|}. \tag{7.2}
$$

#### Step 2: The raw max-margin objective

The margin is the distance to the *closest* point; we maximise it over $\mathbf{w},b$ (Bishop 7.3):
$$
\arg\max_{\mathbf{w},b}\left\{\frac{1}{\|\mathbf{w}\|}\,\min_n\big[t_n(\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)+b)\big]\right\}. \tag{7.3}
$$
The $\frac{1}{\|\mathbf{w}\|}$ factors outside the $\min_n$ because it does not depend on $n$. This is correct but nasty to optimise: a nested max-min, with $\mathbf{w}$ coupling the inner and outer parts.

#### Step 3: Scale invariance rescues us

**Key observation.** The objective is invariant to rescaling $(\mathbf{w},b) \to (\kappa\mathbf{w}, \kappa b)$ for any $\kappa > 0$. The inner $\min_n[\cdot]$ scales by $\kappa$; the $\frac{1}{\|\mathbf{w}\|}$ scales by $\frac{1}{\kappa}$; they cancel. So $(\mathbf{w},b)$ and $(\kappa\mathbf{w},\kappa b)$ describe the **same hyperplane with the same margin** — the solution is an entire equivalence ray, and the value of the inner $\min$ is meaningless on its own (it just reflects the arbitrary scale choice).

We may therefore **fix the scale to a convenient value**. Choose $\kappa$ so the closest point satisfies (Bishop 7.4):
$$
t_n(\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)+b) = 1 \quad\text{(for the closest point)}. \tag{7.4}
$$
We can always do this: if the current minimum is $m>0$, set $\kappa = 1/m$. This is the **canonical representation**.

**Where the inner $\min$ goes — it becomes the constraints.** Once the closest point equals 1 and it *is* the minimum, every other point is automatically $\ge 1$ (Bishop 7.5):
$$
t_n(\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)+b) \ge 1, \qquad n=1,\dots,N. \tag{7.5}
$$
The inner optimisation has been *converted into inequality constraints*. Points where equality holds ($=1$) are **active constraints** = the support vectors lying on the margin hyperplanes $y=\pm1$; points with strict inequality are inactive. There is always $\ge 1$ active constraint, and after maximising the margin, $\ge 2$ (one on each side).

#### Step 4: Max margin becomes a clean minimisation

With the scale fixed, the closest-point distance is $\frac{1}{\|\mathbf{w}\|}$, so the objective is to **maximise** $\frac{1}{\|\mathbf{w}\|}$. We convert via monotonic transforms:
- *Reciprocal:* $\max\frac{1}{\|\mathbf{w}\|} \iff \min\|\mathbf{w}\|$.
- *Square:* squaring is strictly increasing on $[0,\infty)$, so it preserves the minimiser: $\min\|\mathbf{w}\| \iff \min\|\mathbf{w}\|^2$.
- *Factor $\frac{1}{2}$:* a positive constant does not move the minimiser; included so $\nabla\frac{1}{2}\|\mathbf{w}\|^2 = \mathbf{w}$ is clean.

The reason to square: $\|\mathbf{w}\| = \sqrt{\mathbf{w}^\top\mathbf{w}}$ is non-smooth (square root), while $\|\mathbf{w}\|^2 = \mathbf{w}^\top\mathbf{w}$ is a smooth convex quadratic — exactly the form a QP solver wants. Same minimiser, nicer function.

#### The Primal SVM Problem (Bishop 7.6 subject to 7.5)

$$
\boxed{\ \min_{\mathbf{w},b}\ \tfrac{1}{2}\|\mathbf{w}\|^2 \quad\text{subject to}\quad t_n(\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)+b)\ge 1,\ \ n=1,\dots,N.\ } \tag{7.6}
$$

This is a **quadratic program** (QP): convex quadratic objective, linear inequality constraints. Convexity guarantees any local optimum is global. The bias $b$ appears absent from the objective but is determined *implicitly* through the constraints — changes to $\|\mathbf{w}\|$ must be compensated by changes to $b$ to keep the constraints satisfied.

**Summary of the logic chain.** Distance $= \frac{t_n y(\mathbf{x}_n)}{\|\mathbf{w}\|}$ $\;\to\;$ maximise the smallest such distance $\;\to\;$ exploit scale invariance to fix the closest point to 1 (turning the inner min into constraints $\ge 1$) $\;\to\;$ maximising $\frac{1}{\|\mathbf{w}\|}$ becomes minimising $\frac{1}{2}\|\mathbf{w}\|^2$ $\;\to\;$ a convex QP.

Forming the **dual** (next section) is what reveals the kernel formulation and the support-vector sparsity.

### Lagrange Multipliers & KKT Conditions (Bishop App. E, §7.1)

This is both a general optimisation toolkit and its specific application to the SVM primal.

#### Lagrange Multipliers — Equality Constraints (Bishop App. E)

**Problem.** Maximise $f(\mathbf{x})$ subject to $g(\mathbf{x}) = 0$. Form the **Lagrangian**
$$
\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda\, g(\mathbf{x}),
$$
where $\lambda$ is the **Lagrange multiplier**, and find its stationary points ($\nabla_{\mathbf{x}}\mathcal{L} = 0$ and $\partial\mathcal{L}/\partial\lambda = 0$).

**Why it works (Fig. E.1).** The constraint $g(\mathbf{x})=0$ is a $(D{-}1)$-dimensional surface. Two facts:
- $\nabla g$ is **perpendicular** to the constraint surface (gradients point normal to level sets).
- At a constrained optimum, $\nabla f$ must *also* be perpendicular to the surface — if it had any component *along* the surface, one could move along the surface (staying feasible) and increase $f$, contradicting optimality.
![[Pasted image 20260531100500.png]]

Both gradients perpendicular to the same surface $\implies$ they are **parallel**: $\nabla f = -\lambda \nabla g$ for some scalar $\lambda$, i.e. $\nabla f + \lambda \nabla g = 0$. That is exactly $\nabla_{\mathbf{x}}\mathcal{L} = 0$. And $\partial\mathcal{L}/\partial\lambda = 0$ simply re-imposes $g(\mathbf{x})=0$. So stationarity of $\mathcal{L}$ encodes "feasible + gradients aligned."

#### Inequality Constraints & the KKT Conditions (Bishop App. E)

**Problem.** Maximise $f(\mathbf{x})$ subject to $g(\mathbf{x}) \ge 0$. Same Lagrangian $\mathcal{L} = f + \lambda g$, but the analysis splits into **two cases**:

- **Constraint inactive** ($g(\mathbf{x}) > 0$, optimum in the interior): the constraint is not doing anything, so it reduces to an unconstrained max — $\nabla f = 0$ and $\lambda = 0$.
- **Constraint active** ($g(\mathbf{x}) = 0$, optimum on the boundary): like the equality case, $\nabla f = -\lambda \nabla g$. But now the **sign of $\lambda$ matters**: for a maximisation with $g \ge 0$, $\nabla f$ must point into the infeasible direction (opposite $\nabla g$), forcing $\lambda > 0$.

Both cases are captured simultaneously by the **Karush–Kuhn–Tucker (KKT) conditions**:
$$
\begin{aligned}
g(\mathbf{x}) &\ge 0 && \text{(E.9, primal feasibility)} \\
\lambda &\ge 0 && \text{(E.10, dual feasibility)} \\
\lambda\, g(\mathbf{x}) &= 0 && \text{(E.11, complementary slackness)}
\end{aligned}
$$

**Complementary slackness (E.11) is the crucial condition.** The product $\lambda\, g(\mathbf{x}) = 0$ forces *at least one* factor to vanish: if $g > 0$ (inactive) then $\lambda = 0$; if $\lambda > 0$ then $g = 0$ (active). This is exactly the mechanism that produces **sparsity** in the SVM.

![[Pasted image 20260531100650.png]]

#### Applying KKT to the SVM Primal (Bishop 7.7–7.9)

Take the primal (7.6): minimise $\frac{1}{2}\|\mathbf{w}\|^2$ subject to $t_n(\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)+b) \ge 1$. Rewrite each constraint as $g_n := t_n(\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)+b) - 1 \ge 0$, and introduce **one multiplier $a_n \ge 0$ per constraint**. The Lagrangian (Bishop 7.7):
$$
L(\mathbf{w}, b, \mathbf{a}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{n=1}^{N} a_n \big\{ t_n(\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n) + b) - 1 \big\}. \tag{7.7}
$$
The **minus sign** arises because we *minimise* over $(\mathbf{w}, b)$ but *maximise* over $\mathbf{a}$ — the multiplier term is subtracted to enforce the $\ge$ constraints in a minimisation problem.

**Stationarity conditions — differentiate and set to zero:**

Setting $\partial L / \partial \mathbf{w} = 0$: the objective contributes $\mathbf{w}$, the sum contributes $-\sum_n a_n t_n \boldsymbol{\phi}(\mathbf{x}_n)$:
$$
\boxed{\ \mathbf{w} = \sum_{n=1}^{N} a_n\, t_n\, \boldsymbol{\phi}(\mathbf{x}_n)\ } \tag{7.8}
$$
This is the representer form — $\mathbf{w}$ is a linear combination of the (label-signed) training feature vectors, weighted by the multipliers.

Setting $\partial L / \partial b = 0$: only $-\sum_n a_n t_n b$ depends on $b$:
$$
\boxed{\ \sum_{n=1}^{N} a_n\, t_n = 0\ } \tag{7.9}
$$

**The KKT conditions for the SVM (Bishop 7.14–7.16):** applying E.9–E.11 to each constraint gives
$$
a_n \ge 0, \qquad t_n\, y(\mathbf{x}_n) - 1 \ge 0, \qquad a_n\{t_n\, y(\mathbf{x}_n) - 1\} = 0. \tag{7.14–7.16}
$$

The complementary-slackness condition $a_n\{t_n\, y(\mathbf{x}_n) - 1\} = 0$ is the punchline: **for every point, either $a_n = 0$ or $t_n\, y(\mathbf{x}_n) = 1$.**

- **$a_n = 0$:** the point drops out of $\mathbf{w} = \sum_n a_n t_n \boldsymbol{\phi}(\mathbf{x}_n)$ entirely — it plays *no role* in prediction. These are the non-support-vectors.
- **$a_n > 0$:** must have $t_n\, y(\mathbf{x}_n) = 1$, i.e. the point lies *exactly on the margin* hyperplanes $y = \pm 1$. These are the **support vectors**.

This is the origin of sparsity: only the support vectors (points on the margin) contribute to $\mathbf{w}$ and hence to predictions. Substituting (7.8) and (7.9) back into the Lagrangian produces the kernelised dual.

#### The Dual Problem (Bishop 7.10–7.13)

Substitute (7.8) and (7.9) back into the Lagrangian (7.7). The $\mathbf{w}$ and $b$ terms collapse, and the result — after algebra — is expressed entirely in terms of the multipliers:
$$
\widetilde{L}(\mathbf{a}) = \sum_{n=1}^{N} a_n - \frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{N} a_n\, a_m\, t_n\, t_m\, k(\mathbf{x}_n, \mathbf{x}_m) \tag{7.10}
$$
subject to $a_n \ge 0$ and $\sum_n a_n t_n = 0$. This is a **quadratic program in $\mathbf{a}$**, and it depends on the data only through the kernel $k(\mathbf{x}_n, \mathbf{x}_m)$ — the feature map $\boldsymbol{\phi}$ never appears explicitly.

Once the optimal $\mathbf{a}^*$ is found, prediction for a new input uses (7.8) substituted into $y$:
$$
y(\mathbf{x}) = \sum_{n=1}^{N} a_n\, t_n\, k(\mathbf{x}, \mathbf{x}_n) + b. \tag{7.13}
$$
By sparsity, only terms with $a_n > 0$ (support vectors) contribute to this sum.

### Soft-Margin SVM (Bishop §7.1.1)

#### The Problem with Hard Margins

Everything above assumed the data is *linearly separable* in feature space. With overlapping classes that is impossible: no hyperplane satisfies all the constraints, so the hard-margin feasible set is **empty** and the QP has no solution. Even when separation is barely possible, insisting on it lets a single outlier drag the boundary into a terrible position with a tiny margin.

#### Slack Variables (Bishop 7.20)

Introduce one **slack variable** $\xi_n \ge 0$ per point and relax the constraint:
$$
t_n\, y(\mathbf{x}_n) \ge 1 - \xi_n, \qquad \xi_n \ge 0. \tag{7.20}
$$

Three regimes:

| $\xi_n$ value | Meaning |
|---|---|
| $\xi_n = 0$ | Point on or beyond its margin boundary — same as hard margin |
| $0 < \xi_n \le 1$ | Inside the margin but still correctly classified |
| $\xi_n > 1$ | On the wrong side of the decision boundary — **misclassified** |
![[Pasted image 20260531112829.png]]
Since each misclassified point contributes $> 1$ to $\sum_n \xi_n$, this sum is an **upper bound** on the number of misclassifications (not the exact count, because correctly-classified-but-inside-margin points also add their bit).

#### The Soft-Margin Objective (Bishop 7.21)

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}}\ C\sum_{n=1}^{N} \xi_n + \frac{1}{2}\|\mathbf{w}\|^2 \qquad \text{subject to (7.20)}. \tag{7.21}
$$

This is a tradeoff: $\frac{1}{2}\|\mathbf{w}\|^2$ wants a wide margin; $C\sum \xi_n$ penalises violations. The parameter $C > 0$ sets the exchange rate:
- **Large $C$** — violations are expensive, model pushes $\xi_n \to 0$, approaching the hard margin. Low bias, high variance.
- **Small $C$** — violations are cheap, model tolerates them for a fatter margin. More regularised.

Still a convex QP (convex objective, linear constraints), so global-optimum guarantees carry over.

#### Soft-Margin Lagrangian and Stationarity (Bishop 7.22)

Two sets of inequality constraints yield two sets of multipliers: $a_n \ge 0$ for the margin constraints and $\mu_n \ge 0$ for $\xi_n \ge 0$:
$$
L = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_n \xi_n - \sum_n a_n\{t_n\, y(\mathbf{x}_n) - 1 + \xi_n\} - \sum_n \mu_n \xi_n. \tag{7.22}
$$

The $\mathbf{w}$ and $b$ stationarity conditions are **identical** to the hard-margin case:
$$
\frac{\partial L}{\partial \mathbf{w}} = 0 \;\Rightarrow\; \mathbf{w} = \sum_n a_n\, t_n\, \boldsymbol{\phi}(\mathbf{x}_n), \qquad \frac{\partial L}{\partial b} = 0 \;\Rightarrow\; \sum_n a_n\, t_n = 0.
$$

The new condition comes from $\partial L / \partial \xi_n = 0$. Each $\xi_n$ appears in $C\xi_n - a_n \xi_n - \mu_n \xi_n$:
$$
\frac{\partial L}{\partial \xi_n} = C - a_n - \mu_n = 0 \quad\Longrightarrow\quad a_n = C - \mu_n.
$$
Since $\mu_n \ge 0$, this gives $a_n \le C$. Combined with $a_n \ge 0$, the multiplier is now **boxed**:
$$
\boxed{\ 0 \le a_n \le C\ } \tag{7.33}
$$

#### The Soft-Margin Dual (Bishop 7.32–7.34)

Substitute everything back. The slack terms cancel exactly (using $C = a_n + \mu_n$), and $\xi_n$ vanishes from the dual entirely — just as $b$ did. What remains is:
$$
\widetilde{L}(\mathbf{a}) = \sum_n a_n - \frac{1}{2}\sum_n \sum_m a_n\, a_m\, t_n\, t_m\, k(\mathbf{x}_n, \mathbf{x}_m) \tag{7.32}
$$
subject to $0 \le a_n \le C$ (7.33) and $\sum_n a_n t_n = 0$ (7.34).

**The dual objective is identical to the hard-margin dual (7.10).** The *only* structural change is that $a_n \ge 0$ became $0 \le a_n \le C$. All the overlapping-class machinery — slack variables, penalty, second set of multipliers — collapses into a single upper bound on the multipliers. The prediction function (7.13) is unchanged.

#### Three Types of Points Under the Box Constraint

| $a_n$ | $\mu_n = C - a_n$ | Position | Role |
|---|---|---|---|
| $a_n = 0$ | $\mu_n = C$ | On or beyond margin, $\xi_n = 0$ | **Not** a support vector |
| $0 < a_n < C$ | $\mu_n > 0 \;\Rightarrow\; \xi_n = 0$ | Exactly on margin, $t_n y(\mathbf{x}_n) = 1$ | **Margin** support vector |
| $a_n = C$ | $\mu_n = 0 \;\Rightarrow\; \xi_n \ge 0$ | Inside margin or misclassified | **Bound** support vector |

The logic for the middle row: $0 < a_n < C$ means $\mu_n > 0$, and complementary slackness $\mu_n \xi_n = 0$ forces $\xi_n = 0$, so the point sits exactly on the margin. For the bottom row: $a_n = C$ means $\mu_n = 0$, which releases $\xi_n$ to be positive.

**Consequence for recovering $b$:** only the *margin* support vectors ($0 < a_n < C$) satisfy $t_n\, y(\mathbf{x}_n) = 1$, so only they can be used to solve for $b$ via (7.13). Bound support vectors ($a_n = C$) have $t_n\, y < 1$ and would give a wrong $b$.

### Hinge Loss and the Relation to Logistic Regression (Bishop §7.1.1)

#### Rewriting the SVM as Hinge Loss (Bishop 7.44)

Start from the soft-margin objective (7.21). The constraints force $\xi_n \ge 1 - t_n y(\mathbf{x}_n)$ and $\xi_n \ge 0$, and the minimisation pushes each $\xi_n$ as small as possible, so it settles at
$$
\xi_n = \max(0,\; 1 - t_n\, y(\mathbf{x}_n)) = [1 - t_n\, y(\mathbf{x}_n)]_+.
$$
Substituting turns the constrained problem into an **unconstrained** one:
$$
\sum_{n=1}^{N} \underbrace{[1 - t_n\, y(\mathbf{x}_n)]_+}_{E_{\text{SV}}(t_n\, y(\mathbf{x}_n))} + \lambda\|\mathbf{w}\|^2. \tag{7.44}
$$
The function $E_{\text{SV}}(z) = [1 - z]_+$ is the **hinge loss**: zero once correctly classified past the margin ($z \ge 1$), then rising linearly as the point moves toward or past the boundary.

#### The Unifying View — Loss + Regulariser

All three models are "$\sum_n \text{loss} + \frac{\lambda}{2}\|\mathbf{w}\|^2$" with the same L2 regulariser but different losses:

| Model | Loss $E(z)$ where $z = t_n\, y(\mathbf{x}_n)$ |
|---|---|
| Linear regression | $\{t_n - \mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)\}^2$ (squared error) |
| Logistic regression | $\log(1 + \exp(-t_n\, \mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)))$ (cross-entropy) |
| SVM | $[1 - t_n\, \mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)]_+$ (hinge) |

#### Comparing the Losses (Figure 7.5)

![[Pasted image 20260531123551.png]]
Plot all losses against the margin quantity $z = t_n\, y(\mathbf{x}_n)$:

- **Hinge** and **log loss** are remarkably similar — both are convex upper bounds on the 0/1 misclassification step function, both penalise being on the wrong side heavily and reward confident correctness. An SVM and L2-regularised logistic regression are solving almost the same optimisation with slightly different loss shapes, so they usually give similar decision boundaries.
- **Hinge has a flat zero for $z \ge 1$** — once a point is past the margin it contributes *exactly nothing*. This flat region is the source of SVM sparsity: those points get $a_n = 0$. Log loss *never* reaches exactly zero — every point keeps contributing, so logistic regression has no sparsity.
- **Hinge has a kink at $z = 1$** (non-differentiable), requiring QP/subgradient methods rather than the smooth gradient descent logistic regression enjoys.
- **Log loss gives probabilities** ($\sigma(y)$ is a calibrated $p(t{=}1|\mathbf{x})$); hinge gives only a decision — this is the limitation that SVMs do not output posterior probabilities.
- **Squared loss** increases again for $z > 1$, penalising points for being "too correct" — this is why squared loss is unsuitable for classification.

### SVM for Regression (Bishop §7.1.2)

#### The $\epsilon$-Insensitive Error Function (Bishop 7.51)

Ordinary regression uses squared loss — every deviation is penalised. Support vector regression replaces it with the **$\epsilon$-insensitive loss**:
$$
E_\epsilon(y(\mathbf{x}) - t) = \begin{cases} 0 & \text{if } |y(\mathbf{x}) - t| < \epsilon \\ |y(\mathbf{x}) - t| - \epsilon & \text{otherwise} \end{cases} \tag{7.51}
$$
The flat zero region creates a **tube** of tolerance of width $\pm\epsilon$ — predictions within $\epsilon$ of the target cost nothing. This flat region produces **sparsity**: points inside the tube have zero loss and will not be support vectors.
![[Pasted image 20260531123814.png]]

The objective is the familiar loss + regulariser (Bishop 7.52):
$$
C\sum_{n=1}^{N} E_\epsilon(y(\mathbf{x}_n) - t_n) + \frac{1}{2}\|\mathbf{w}\|^2. \tag{7.52}
$$

#### Why Slack Variables? The Hinge-Loss Trick in Reverse

In classification (slide 29), we went from constrained-with-slack to unconstrained-with-loss by eliminating $\xi_n = [1 - t_n y_n]_+$. Constrained-with-slack and unconstrained-with-loss are **two encodings of the same problem** — the slack variable *is* the loss, lifted into an extra dimension so the kink becomes linear.

Regression starts from the other encoding: we begin with $E_\epsilon$ and to feed it to a QP solver we must *introduce* slack variables to linearise it. The reason is mechanical: $E_\epsilon$ has a non-differentiable kink (like the hinge), and QP solvers want a smooth quadratic objective with *linear* constraints. The slack variable absorbs the kink:
$$
|y_n - t_n| - \epsilon \;\text{ (outside tube)} \;=\; \min(\xi_n + \hat{\xi}_n) \;\text{ s.t. } \begin{cases} y_n - t_n \le \epsilon + \xi_n \\ t_n - y_n \le \epsilon + \hat{\xi}_n \\ \xi_n, \hat{\xi}_n \ge 0 \end{cases}
$$
The left side is a kinked loss; the right side is two linear inequalities plus nonnegativity — exactly the ingredients of a QP. We trade "one variable with an awkward $|\cdot|$ and a flat spot" for "two well-behaved nonnegative variables with linear constraints."

#### Two Sets of Slack Variables

In classification, a point can only violate the margin from one side. In regression, a point can fall outside the tube either **above** or **below** — these are mutually exclusive, so two slack variables per point are needed:
- $\xi_n \ge 0$ — distance above the tube ($y(\mathbf{x}_n) > t_n + \epsilon$).
- $\hat{\xi}_n \ge 0$ — distance below the tube ($y(\mathbf{x}_n) < t_n - \epsilon$).
- Inside the tube: both zero.
![[Pasted image 20260531123829.png]]

The objective becomes $C\sum_n(\xi_n + \hat{\xi}_n) + \frac{1}{2}\|\mathbf{w}\|^2$. After the full Lagrangian/dual treatment with two multiplier sets $a_n, \hat{a}_n$, the dual is (Bishop 7.61):
$$
\widetilde{L}(\mathbf{a}, \hat{\mathbf{a}}) = -\frac{1}{2}\sum_n \sum_m (a_n - \hat{a}_n)(a_m - \hat{a}_m)\, k(\mathbf{x}_n, \mathbf{x}_m) - \epsilon\sum_n (a_n + \hat{a}_n) + \sum_n (a_n - \hat{a}_n)\, t_n. \tag{7.61}
$$
The multipliers always appear as the **difference** $(a_n - \hat{a}_n)$ — a point cannot be above *and* below simultaneously, so at most one of each pair is nonzero. This difference plays the role that $a_n t_n$ played in classification.

#### Prediction and Support Vectors (Bishop 7.64)

$$
y(\mathbf{x}) = \sum_{n=1}^{N} (a_n - \hat{a}_n)\, k(\mathbf{x}, \mathbf{x}_n) + b. \tag{7.64}
$$
The support vectors are precisely the points **on or outside the $\epsilon$-tube**. Points inside the tube have $a_n = \hat{a}_n = 0$ and do not enter the sum — the same sparsity story as the classifier.

#### The Unifying Picture

Across the whole chapter, one template recurs: **loss $+ \frac{1}{2}\|\mathbf{w}\|^2$**, where a loss with a *flat zero region* (hinge for classification, $\epsilon$-insensitive for regression) produces sparse solutions via KKT complementary slackness, and the dual expresses everything through kernels. Classification needs one slack/multiplier per point (violation is one-directional); regression needs two (the tube can be breached from either side).

