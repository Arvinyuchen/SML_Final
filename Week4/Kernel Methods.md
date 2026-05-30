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

