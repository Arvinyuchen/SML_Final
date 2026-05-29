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