**Goal**: map input vector $\mathbf{x}$ to one of $K$ discrete classes $\mathcal{C}_k$. 
**One-hot encoding**: $\mathbf{t}$ a length $K$ vector, each $t_k$ is interpreted as $p(\mathcal{C_k})$.

There are two approaches to determine the conditional probabilities $p(C_k \mid \mathbf{x})$. 
- Model them directly, for example by representing them as parametric models and then optimizing the parameters using a training set.
- Alternatively, we can adopt a generative approach in which we model the class-conditional densities given by $p(C_k \mid \mathbf{x})$, together with the prior probabilities $p(\mathcal{C}_k)$ for the classes, and then we compute the required posterior probabilities using Bayes' theorem $$
p(\mathcal{C_k} \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \mathcal{C}_k) p(\mathcal{C}_k)}{p(\mathbf{x})}$$
**Generalised linear model.** Regression used $y(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + w_0$. For classification we wrap it in a nonlinearity:
$$
y(\mathbf{x}) = f(\mathbf{w}^T\mathbf{x} + w_0).
$$
$f(\cdot)$ is the activation function
## Decision Theory
### Minimising the misclassification rate
In a classification problem, dividing the input space into regions $\mathcal{R}_k$ (*decision regions*), such that all points in $\mathcal{R}_k$ are assigned to class $\mathcal{C_k}$ is a decision rule. The boundaries between the decision regions are called *decision boundaries* or *decision surfaces*. 

E.g., cancer problem, a mistake occurs when an input vector belonging to class $\mathcal{C}_1$ is assigned to class $\mathcal{C}_2$ or vice versa. The probability of this occurring is given by
$$
\begin{aligned}  
p(\text{mistake})  
&= p(\mathbf{x} \in \mathcal{R}_1, \mathcal{C}_2)  
+ p(\mathbf{x} \in \mathcal{R}_2, \mathcal{C}_1) \\  
&= \int_{\mathcal{R}_1} p(\mathbf{x}, \mathcal{C}_2)\, d\mathbf{x}  
+ \int_{\mathcal{R}_2} p(\mathbf{x}, \mathcal{C}_1)\, d\mathbf{x}.  
\end{aligned}
$$We are free to choose the decision rule that assigns each point $\mathbf{x}$ to one of the two classes. Clearly to minimise  $p(\text{mistake})$ we should arrange that each $\mathbf{x}$ is assigned to whichever class has the smaller value for the integrated. By product rule $p(\mathbf{x}, \mathcal{C}_2) = p(\mathcal{C}_k \mid \mathbf{x}) p(\mathbf{x})$ we can rewrite  
$$
\int_{\mathcal{R}_1} p(\mathcal{C}_1 \mid \mathbf{x})p(\mathbf{x}) d\mathbf{x}  
+ \int_{\mathcal{R}_2} p(\mathcal{C}_2 \mid \mathbf{x})p(\mathbf{x}) d\mathbf{x}.
$$
The minimum probability of making a mistake is obtained if each value of $\mathbf{x}$ is assigned to the class for which the posterior probability $p(\mathcal{C}_k \mid \mathbf{x})$ is largest, illustrated in the Figure 1.24.
![[Pasted image 20260527111457.png]]
### Minimising the expected loss
However, the consequences of these two types of mistake can be dramatically different. We then introduce *loss function* which is a single, overall measure of loss incurred in taking any of the available decisions or actions, and our goal is to minimise the loss. However, the true loss function depends on the true class, which is unknown for a new sample $\mathbf{x}$, our uncertainty in the true class is expressed through the joint probability distribution $p(\mathbf{x}, \mathcal{C}_k)$ and so we seek instead to minimise the average loss, where the average is computed with respect to this distribution, which is given by
$$
\mathbb{E} [L] = \sum_k \sum_j \int_{\mathcal{R}_j} L_{kj}\ p (\mathbf{x}, \mathcal{C}_k) d\mathbf{x}.
$$
Each $\mathbf{x}$ can be assigned independently to one of the decision regions $\mathcal{R}_j$. Our goal is to choose the regions $\mathcal{R}_j$ in order to minimise the expected loss, which implies that for each $\mathbf{x}$ we should minimise $\sum_k L_{kj} p(\mathbf{x}, \mathcal{C}_k)$. Apply the product rule again, the decision rule that minimises the expected loss is the one that assigns each new $\mathbf{x}$ to the class $j$ for which the quantity 
$$
\sum _k L_{kj}\ p(\mathcal{C}_k \mid \mathbf{x})
$$
is minimum.
### The reject option
We have seen that classification errors arise from the regions of input space where the largest of the posterior probabilities $p(\mathcal{C}_k \mid \mathbf{x})$ is significantly less than 1, or equivalently where the joint distributions $p(\mathbf{x}, \mathcal{C}_k)$ have comparable values. These are the regions where we are relatively uncertain about class membership, and in some applications it will be appropriate to avoid making decisions on the difficult cases in anticipation of a lower error rate on those examples for which a classification decision is made.
![[Pasted image 20260527114910.png]]
### Inference and decision
We have broken the classification problem down into
- *inference stage* in which we use training data to lean a model for $p(\mathcal{C}_k \mid \mathbf{x})$,
- and the subsequent *decision* stage in which we use there posterior probabilities to make optimal class assignments.
Alternatively, a *discrimination function* that maps inputs $\mathbf{x}$ directly into decisions can be learned.
We can identify **three distinct approaches** to solving decision problems
- First solve the inference problem of determining the class-conditional densities $p(\mathbf{x}\mid \mathcal{C}_k)$ for each class $\mathcal{C}_k$ individually. Also separately infer the prior class probabilities $p(\mathcal{C}_k)$. Then use Bayes' theorem in the form $$ p (\mathcal{C_k} \mid \mathbf{x}) = \frac{p (\mathbf{x}\mid \mathcal{C}_k) p(\mathcal{C}_k)}{p(\mathbf{x})} $$to find the posterior class probabilities $p(\mathcal{C}_k \mid \mathbf{x})$. As usual, the denominator in Bayes' theorem can be found in terms of the quantities appearing in the numerator because $$ p(\mathbf{x}) = \sum_k p(\mathbf{x}\mid \mathcal{C}_k) p(\mathcal{C}_k) . $$ Equivalently, we can model the joint distribution $p(\mathbf{x}, \mathcal{C}_k)$ directly and then normalise to obtain the posterior probabilities. Having found the posterior probabilities, we use decision theory to determine class membership for each new input $\mathbf{x}$. Approaches that explicitly or implicitly model the distribution of inputs as well as outputs are known as _generative models_, because by sampling from them it is possible to generate synthetic data points in the input space.
- First solve the inference problem of determining the posterior class probabilities $p(\mathcal{C}_k|\mathbf{x})$, and then subsequently use decision theory to assign each new $\mathbf{x}$ to one of the classes. Approaches that model the posterior probabilities directly are called _discriminative models_.
- Find a function $f(x)$, called a discriminant function, which maps each input $\mathbf{x}$ directly onto a class label.

Approach (a) is the most demanding, it involves finding the joint distribution over $\mathbf{x}$ and $\mathcal{C}_k$. However, it allows the marginal density of data $p(\mathbf{x})$ to be determined. This can be useful for detecting new data points that how low probability under the model and for which the predictions may be of low accuracy, which is known as *outlier detection* or *novelty detection*. 

Approach (b) can directly obtain the posterior probabilities that can be used to make classification decisions saving computational resources to find the joint distribution $p(\mathbf{x}, \mathcal{C}_k)$ when the class- conditional densities may contain a lot of structure that has little effect on the posterior probabilities, as illustrated in Figure 1.27.
![[Pasted image 20260527173626.png]]
Approach (c) combines the inference and decision stages into a single learning problem. In the example of Figure 1.27, this would correspond to finding the value of x shown by the vertical green line, because this is the decision boundary giving the minimum probability of misclassification. However, we no longer have access to the posterior probabilities $p(\mathcal{C}_k \mid \mathbf{x})$. 

## Discriminant Functions
### Two classes
$$  
y(\mathbf{x})  
=  
\underbrace{\mathbf{w}^{\mathrm T}}_{\text{weight vector}}  
\mathbf{x}  
+  
\underbrace{w_0}_{\text{bias}},  
\qquad  
\mathcal{C}_1 \text{ if } y(\mathbf{x}) \ge 0,  
\quad  
\mathcal{C}_2 \text{ otherwise.}  
\tag{4.4}  
$$
- $\mathbf{w}$ is orthogonal to the decision surface
- the bias $w_0$ sets the offset from the origin
- the value of $y(\mathbf{x})$ gives a signed measure of the perpendicular distance $r$ of the point $\mathbf{x}$ from the decision surface. To see this, consider an arbitrary point $\mathbf{x}$ and let $\mathbf{x}_\perp$ be its orthogonal projection onto the decision surface so that $$ \mathbf{x} = \mathbf{x}_\perp + r \frac{\mathbf{w}}{\| \mathbf{w} \|}.$$ Multiplying both sides of this result by $\mathbf{w}^T$ and adding $w_0$, and making use of $y(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + w_0$ and $y(\mathbf{x}_\perp) = \mathbf{w}^T\mathbf{x}_\perp + w_0 = 0$, we have $$r = \frac{y(\mathbf{x})}{\|\mathbf{w}\|}.$$![[Pasted image 20260527173701.png]]
### Multiple classes
Consider the use of $K-1$ classifiers each of which solves a two-class problem of separating points in a particular class $\mathcal{C}_k$ from points not in that class. 
- The left-hand example in Figure 4.2 shows an example involving three classes where this approach leads to regions of input space that are ambiguously classified.
-  $K(K-1)/2$ binary discriminant functions, one for every possible pair of classes. Each point is then classified according to a majority vote amongst the discriminant functions, but also runs into the ambiguous regions, right-hand diagram of Figure 4.2.
![[Pasted image 20260527125705.png]]
We can avoid these difficulties by considering a single $K$-class discriminant comprising $K$ linear functions of the form 
$$
y_k(\mathbf{x}) = \mathbf{w}_k^T \mathbf{x} + w_{k0}
$$
and then assigning a point $\mathbf{x}$ to class $\mathcal{C}_k$ if $y_k(\mathbf{x}) > y_j(\mathbf{x})$ for all $j \neq k$. 
![[Pasted image 20260527132321.png]]
The decision boundary between class $\mathcal{C}_k$ and class $\mathcal{C}_j$ is therefore given by $y_k(\mathbf{x}) = y_j(\mathbf{x})$ and hence corresponds to a $(D-1)$-dimensional hyperplane defined by 
$$
(\mathbf{w}_k - \mathbf{w}_j)^T \mathbf{x} + (w_{k0}-w_{j0}) = 0.
$$
The resulting decision regions are **singly connected and convex**.
### Least squares for classification
Each class $\mathcal{C}_k$ is described by its own linear model so that
$$
y_k(\mathbf{x}) = \mathbf{w}_k^T \mathbf{x} + w_{k0}
$$
where $k= 1,\dots, K$. We can conveniently group these together using vector notations so that
$$
\mathbf{y}(\mathbf{x}) = \widetilde{\mathbf{W}}^T \tilde{\mathbf{x}}
$$
where $\widetilde{W}$, is a matrix whose $k^{th}$ column comprises the $D+1$-dimensional vector $\tilde{w}_k =(w_{k0}, \mathbf{w}^T_k)^T$  and $\tilde{x}$ is the corresponding augmented input vector $(1, \mathbf{x}^T)^T$ with a dummy input $x_0=1$. A new input $\mathbf{x}$ is then assigned to the class for which the output $y_k = \tilde{\mathbf{w}}^T \tilde{\mathbf{x}}$ is largest.

We now determine the parameter matrix $\widetilde{W}$ by minimising a sum-of-squares error function. Consider a training data set $\{ \mathbf{x}_n, \mathbf{t}_n \}$ when $n=1, \dots, N$, and define a matrix $\mathbf{T}$ whose $n^{th}$ row is the vector $\mathbf{t}_n^T$ , together with a matrix $\tilde{\mathbf{X}}$ whose $n^{th}$ row is $\tilde{x}_n^T$. The sum-of-squares error function can then be written as
$$
E_D(\widetilde{W}) = \frac{1}{2} \text{Tr}\left\{ (\widetilde{\mathbf{X}} \widetilde{\mathbf{W}} - \mathbf{T})^T (\widetilde{\mathbf{X}} \widetilde{\mathbf{W}} - \mathbf{T}) \right\}.
$$
Setting the derivative with respect to $\widetilde{W}$ to zero, and rearranging, we then obtain the solution for $\widetilde{\mathbf{W}}$ in the form
$$
\widetilde{\mathbf W}=(\widetilde{\mathbf X}^{\mathrm T}\widetilde{\mathbf X})^{-1}\widetilde{\mathbf X}^{\mathrm T}\mathbf T=\widetilde{\mathbf X}^{\dagger}\mathbf T. \tag{4.16}
$$
![[Pasted image 20260527142246.png]]

## Probabilistic Generative Models
Take Bayes' theorem for two classes and just rewrite it
$$ p(\mathcal C_1\mid\mathbf x)=\frac{p(\mathbf x\mid\mathcal C_1)p(\mathcal C_1)}{p(\mathbf x\mid\mathcal C_1)p(\mathcal C_1)+p(\mathbf x\mid\mathcal C_2)p(\mathcal C_2)}=\frac{1}{1+\exp(-a)}=\sigma(a), \tag{4.57}$$
where we have defined 
$$ a(\mathbf x)=\ln\frac{p(\mathbf x\mid\mathcal C_1)p(\mathcal C_1)}{p(\mathbf x\mid\mathcal C_2)p(\mathcal C_2)}=\ln\frac{p(\mathbf x,\mathcal C_1)}{p(\mathbf x,\mathcal C_2)}. \tag{4.58}$$
![[Pasted image 20260527143526.png]]
 Properties:
- Symmetry: $\sigma(-a)=1-\sigma(a)$.
- Inverse = logit: $a=\ln\frac{\sigma}{1-\sigma}=\ln\frac{p(\mathcal C_1\mid\mathbf x)}{p(\mathcal C_2\mid\mathbf x)}$.

For the case of $K > 2$ classes, we have
$$
\begin{aligned}
p(\mathcal{{C}_k}\mid \mathbf{x}) 
&= \frac{p(\mathbf{x}\mid \mathcal{C}_k ) p(\mathcal{C}_k)}{\sum_j (\mathbf{x}\mid \mathcal{C}_j ) p(\mathcal{C}_j)} \\
&= \frac{\exp(a_k)}{\sum_j \exp(a_j)}
\end{aligned}
\tag{4.62}
$$
which is known as the *normalised exponential* or *softmax function* and can be regarded as a multi-class generalisation of the logistics sigmoid. Here the quantities $a_k$ are defined by
$$
a_k = \ln p(\mathbf{x} \mid \mathcal{C}_k)p(\mathcal{C}_k) \tag{4.63}.
$$
### Continuous inputs
Let us assume that the class-conditional densities are Gaussian and then explore the resulting form for the posterior probabilities. To start with, we shall assume that all classes share the same covariance matrix. Thus the density for class $\mathcal{C}_k$ is given by
$$ p(\mathbf x\mid\mathcal C_k)=\frac{1}{(2\pi)^{D/2}|\boldsymbol\Sigma|^{1/2}}\exp\!\Big\{-\tfrac12(\mathbf x-\boldsymbol\mu_k)^{\mathrm T}\boldsymbol\Sigma^{-1}(\mathbf x-\boldsymbol\mu_k)\Big\}. \tag{4.64}$$

Consider first the case of two classes. From (4.57) and (4.58), we have
$$
p(\mathcal C_1\mid\mathbf x)=\sigma(\mathbf w^{\mathrm T}\mathbf x+w_0), \tag{4.65}
$$
where we have defined
$$
\mathbf{w} = \mathbf{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)
\tag{4.66}
$$
$$
w_0
= -\frac{1}{2}\boldsymbol{\mu}_1^{T}\mathbf{\Sigma}^{-1}\boldsymbol{\mu}_1
+ \frac{1}{2}\boldsymbol{\mu}_2^{T}\mathbf{\Sigma}^{-1}\boldsymbol{\mu}_2
+ \ln \frac{p(\mathcal{C}_1)}{p(\mathcal{C}_2)}.
\tag{4.67}
$$
This result is illustrated for the case of a two-dimensional input space $\mathbf{x}$ in Figure 4.10.
![[Pasted image 20260527154921.png]]
If we relax the assumption of a shared covariance matrix and allow each class conditional density $p(\mathbf{x} \mid \mathcal{C}_k)$ to have its own covariance matrix $\Sigma_k$, then the earlier cancellations will no longer occur, and we will obtain quadratic functions of $\mathbf{x}$, giving rise to a *quadratic discriminant*. The linear and quadratic decision boundaries are illustrated in Figure 4.11.
![[Pasted image 20260527155314.png]]
### Maximum likelihood solution
Once we have specified a parametric functional form for the class-conditional densities $p(\mathbf{x}\mid \mathcal{C}_k)$,we can then determine the values of the parameters, together with the prior class probabilities $p(\mathcal{C}_k)$, using maximum likelihood. This requires a data set comprising observations of $\mathbf{x}$ along with their corresponding class labels. 

Two classes where each having a Gaussian class-conditional density with a shared covariance matrix, and suppose we have a data set $\{\mathbf{x}_n, t_n\}$ where $n = 1 , \dots, N$. Here $t_n = 1$ denotes class $\mathcal{C}_1$ and $t_n = 0$ denotes class $\mathcal{C}_2$. We denote the prior class probability $p(\mathcal{C}_1) =\pi$, so that $p(\mathcal{C}_2) = 1- \pi$. For a data point $\mathbf{x}_n$ from class $\mathcal{C}_1$, we have $t_n = 1$ and hence
$$
p(\mathbf{x}_n , \mathcal{C}_1) = p(\mathcal{C}_1) p(\mathbf{x}_n \mid \mathcal{C}_1) = \pi \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_1 , \mathbf{\Sigma}).
$$
Similarly for class $\mathcal{C}_2$, we have $t_n = 0$ and hence
$$
p(\mathbf{x}_n , \mathcal{C}_2) = p(\mathcal{C}_2) p(\mathbf{x}_n \mid \mathcal{C}_2) = (1-\pi) \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_2 , \mathbf{\Sigma}).
$$
Thus the likelihood function is given by
$$
p(\mathbf{t}\mid \pi,\boldsymbol{\mu}_1,\boldsymbol{\mu}_2,\boldsymbol{\Sigma})  
=  
\prod_{n=1}^{N}  
\left[  
\pi \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_1,\boldsymbol{\Sigma})  
\right]^{t_n}  
\left[  
(1-\pi)\mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_2,\boldsymbol{\Sigma})  
\right]^{1-t_n}
$$ where $\mathbf{t} = (t_1, \dots, t_N)^T.$ As usual, it is convenient to maximize the log of the likelihood function. 
#### Maximisation with respect to $\pi$

The terms in the log likelihood function that depend on $\pi$ are
$$
\sum^N_{n=1} \{ t_n \ln \pi + (1-t_n) \ln(1-\pi)\}.
$$
Setting the derivative with respect to $\pi$ equal to zero and rearranging, we obtain
$$
\pi = \frac{1}{N} \sum^N_{n=1}t_n = \frac{N_1}{N} = \frac{N_1}{N_1+N_2}
$$
where $N_1$ denotes the total number of data points in class $\mathcal{C}_1$, and $\mathcal{C}_1$ and $N_2$ denotes the total number of data points in class $\mathcal{C}_2$. 
#### Maximisation with respect to $\mu_1$
Again we can pick out of the log likelihood function those terms that depend on $\mu_1$ giving
$$
\sum^N_{n=1} t_n \ln \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_1, \mathbf{\Sigma}) = -\frac{1}{2} \sum^N_{n=1} t_n (\mathbf{x}_n - \boldsymbol{\mu_1}) ^T \mathbf{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu_1} ) + \text{const}.
$$
Setting the derivative with respect to $\boldsymbol{\mu}_1$ to zeo and rearranging, we obtain
$$
\boldsymbol{\mu}_1 = \frac{1}{N_1} \sum^N_{n=1}t_n \mathbf{x}_n
$$
which is simply the mean of all the input vectors $\mathbf{x}_n$ assigned to class $\mathcal{C}_1$. By a similar argument, the corresponding result for $\boldsymbol{\mu}_2$ is given by
$$
\boldsymbol{\mu_2} = \frac{1}{N_2 }\sum^N_{n=1} (1-t_n)\mathbf{x}_n. 
$$
Finally, consider the maximum likelihood solution for the shared covariance matrix $\mathbf{\Sigma}$. Picking out the terms in the log likelihood function that depend on $\mathbf{\Sigma}$, we have
$$

\begin{aligned}
&-\frac{1}{2}\sum_{n=1}^{N} t_n \ln |\boldsymbol{\Sigma}|
-\frac{1}{2}\sum_{n=1}^{N} t_n
(\mathbf{x}_n-\boldsymbol{\mu}_1)^{\mathrm{T}}
\boldsymbol{\Sigma}^{-1}
(\mathbf{x}_n-\boldsymbol{\mu}_1)
\\
&
-\frac{1}{2}\sum_{n=1}^{N} (1-t_n)\ln |\boldsymbol{\Sigma}|
-\frac{1}{2}\sum_{n=1}^{N} (1-t_n)
(\mathbf{x}_n-\boldsymbol{\mu}_2)^{\mathrm{T}}
\boldsymbol{\Sigma}^{-1}
(\mathbf{x}_n-\boldsymbol{\mu}_2)
\\
&=
-\frac{N}{2}\ln |\boldsymbol{\Sigma}|
-\frac{N}{2}\operatorname{Tr}
\left\{
\boldsymbol{\Sigma}^{-1}\mathbf{S}
\right\}.
\end{aligned}
\tag{4.77}
$$
where we have defined
$$
\begin{align}
\mathbf{S}
&=
\frac{N_1}{N}\mathbf{S}_1
+
\frac{N_2}{N}\mathbf{S}_2
\tag{4.78}
\\[0.5em]
\mathbf{S}_1
&=
\frac{1}{N_1}
\sum_{n\in \mathcal{C}_1}
(\mathbf{x}_n-\boldsymbol{\mu}_1)
(\mathbf{x}_n-\boldsymbol{\mu}_1)^{\mathrm{T}}
\tag{4.79}
\\[0.5em]
\mathbf{S}_2
&=
\frac{1}{N_2}
\sum_{n\in \mathcal{C}_2}
(\mathbf{x}_n-\boldsymbol{\mu}_2)
(\mathbf{x}_n-\boldsymbol{\mu}_2)^{\mathrm{T}} .
\tag{4.80}
\end{align}
$$
### Discrete features
Make the *naive Bayes* assumption in which the feature values are treated as independent, conditioned on the class $\mathcal{C}_k$. Thus we have class-conditional distributions of the form
$$
p(\mathbf{x}\mid \mathcal{C}_k) = \prod^D_{i=1} \mu_{ki}^{x_i} (1-\mu_{ki})^{1-x_i}
\tag{4.81}
$$
which contain $D$ independent parameters for each class. Substituting into (4.63) then gives
$$
a_k(\mathbf{x})
=
\sum_{i=1}^{D}
\left\{
x_i \ln \mu_{ki}
+
(1-x_i)\ln(1-\mu_{ki})
\right\}
+
\ln p(C_k)
\tag{4.82}
$$
which again are linear functions of the input values $x_i$. 

## Probabilistic Discriminative Models
An alternative approach is to use the functional form of the generalized linear model explicitly and to determine its parameters directly by using maximum likelihood.
- Indirect approach to finding the parameters of a generalised linear model by fitting class-conditional densities and class priors separately and then applying Bayes' theorem (generative). 
- In direct approach, we are maximising a likelihood function defined through the conditional distribution $p (\mathcal{C}_k \mid \mathbf{x})$ (discriminative). Pros: fewer adaptive parameters to be determined. Improved predictive performance particularly when the class-conditional density assumptions give a poor approximation to the true distributions.

### Fixed basis functions
The application of basis functions $\boldsymbol{\phi}$ on the original input vector $\mathbf{x}$ would result decision boundaries be linear in the feature space $\boldsymbol{\phi}$, and thee corresponds to nonlinear decision boundaries in the original $\mathbf{x}$ space, as illustrated in Figure 4.12.
![[Pasted image 20260527190615.png]]

### Logistics regression 
Under rather general assumptions, the posterior probability of class $\mathcal{C}_1$ can be written as a logistic sigmoid acting on a linear function of the feature vector $\phi$ so that
$$
p(\mathcal{C}_1 \mid \boldsymbol{\phi}) = y(\boldsymbol{\phi})
 = \sigma (\mathbf{w}^T\boldsymbol{\phi})$$
 with $p(\mathcal{C}_2 \mid \boldsymbol{\phi}) = 1 - p(\mathcal{C}_1 \mid \boldsymbol{\phi})$. 
We now use maximum likelihood to determine the parameters of the logistic regression model. We will be using the derivative of the logistics sigmoids function which is neat
$$
\frac{d\sigma}{d a} = \sigma(1-\sigma). \tag{4.88}
$$
For a data set $\{\phi_n, t_n\}$, where $t_n \in \{0,1\}$ and $\phi_n = \phi(\mathbf{x}_n)$, with $n = 1, \ldots, N$, the likelihood function can be written
$$
p(\mathbf{t}\mid \mathbf{w})
=
\prod_{n=1}^{N}
y_n^{t_n}
\{1-y_n\}^{1-t_n}
\tag{4.89}
$$
where
$$
\mathbf{t} = (t_1,\ldots,t_N)^{\mathrm{T}},\quad y_n = p(\mathcal{C}_1 \mid \phi_n).
$$
As usual, we can define an error function by taking the negative logarithm of the likelihood, which gives the cross-entropy error function in the form
$$
E(\mathbf{w})
=
-\ln p(\mathbf{t}\mid \mathbf{w})
=
-\sum_{n=1}^{N}
\left\{
t_n \ln y_n
+
(1-t_n)\ln(1-y_n)
\right\}
\tag{4.90}
$$
where
$$
y_n = \sigma(a_n), \quad a_n = \mathbf{w}^{\mathrm{T}}\phi_n.
$$
Taking the gradient of the error function with respect to $\mathbf{w}$, we obtain
$$
\nabla E(\mathbf{w})
=
\sum_{n=1}^{N}
(y_n - t_n)\phi_n
\tag{4.91}
$$
 where we have made use of (4.88).

## The Laplace Approximation
In next section we will discuss the Bayesian treatment of logistics regression. We cannot integrate exactly over the parameter vector $\mathbf{w}$ since the posterior distribution is no longer Gaussian. It is therefore necessary to introduce some form of approximation. 

Laplace approximation is a simple, but widely used framework that aims to find a Gaussian approximation to a probability density defined over a set of continuous variables. Consider first the case of a single continuous variable $z$, and suppose the distribution $p(z)$ is defined by 
$$
p(z) = \frac{1}{Z} f(z)
$$
where $Z = \int f(z)\ dz$ is the normalisation coefficient. In the Laplace method the goal is to find a Gaussian approximation $q(z)$ which is centred on a mode of the distribution $p(z)$. The first step is to find a mode of $p(z)$, in other words a point $z_0$ such that $p'(z_0) = 0$, or equivalently
$$
\left.
\frac{df(z)}{dz}
\right|_{z=z_0}
= 0
$$
A Gaussian distribution has the property that its logarithm is a quadratic function of the variables. We therefore consider a Taylor expansion of $\ln f(z)$. We therefore consider a Taylor expansion of $\ln f(z)$ centred on the mode $z_0$, a second-order Taylor expansion of $\ln f(z)$ around $z_0$ is, in general
$$
\ln f(z)
\simeq
\ln f(z_0)
+ \underbrace{\left . \frac{d\ln f}{dz} \right|_{z_0} }_{\text{1st-order term}}(z-z_0) + \left . \frac{1}{2} \frac{d^2 \ln f}{dz^2} \right|_{z_0} (z-z_0)^2.
$$
The first-order term vanishes since $z_0$ is a local maximum of the distribution. We then use 
$$
A = - \left . \frac{d^2}{dz^2} \ln f(z) \right|_{z=z_{0}}
$$
to simplify the equation to
$$
\ln f(z) \simeq f(z_{0})-\frac{1}{2}A(z-z_{0})^2.
$$
Taking the exponential we obtain
$$
f(z)\simeq f (z_{0}) \exp \left\{ -\frac{A}{2} (z-z_{0})^2 \right\}.
$$
We can then obtain a normalised distribution $q(z)$ by making use of the standard result fro the normalisation of a Gaussian, so that 
$$
q(z)=\left(\frac{A}{2\pi}\right)^{1/2} \exp \left\{ - \frac{A}{2 } (z-z_{0})^2 \right\}.
$$
The Laplace approximation is illustrated in Figure 4.14.
![[Pasted image 20260527205756.png]]
Note that the Gaussian approximation will only be well defined if its precision $A >0$, in other words the stationary point $z_0$ must be a local maximum, so that the second derivative of $f(z)$ at the point $z_0$ is negative.

We can extend the Laplace method to approximate a distribution $p(\mathbf{z})=\frac{f(\mathbf{z})}{Z}$ defined over an $M$-dimensional space $\mathbf{z}$. At stationary point $\mathbf{z_{0}}$ the gradient $\nabla f(\mathbf{z})$ will vanish. Expanding around this stationary point we have
$$
\ln f(\mathbf{z}) \simeq \ln f(\mathbf{z_{0}})-\frac{1}{2}(\mathbf{z-\mathbf{z_{0}}})^T \mathbf{A}(\mathbf{z-\mathbf{z_{0}}})
$$
where the $M\times M$ Hessian matrix $\mathbf{A}$ is defined by
$$
\mathbf{A=-\nabla \nabla \left . \ln f(\mathbf{z}) \right|_{\mathbf{z}=\mathbf{z}_{0}}}
$$
and $\nabla$is the gradient operator. Taking the exponential of both sides we obtain
$$
f(\mathbf{z})\simeq f(\mathbf{z}_{0})\exp \left\{ -\frac{1}{2}(\mathbf{z}-\mathbf{z}_{0})^T \mathbf{A}(\mathbf{z}-\mathbf{z}_{0}) \right\}.
$$
The distribution $q(\mathbf{z})$ is proportional to $f(\mathbf{z})$ and the appropriate normalisation coefficient can be found by inspection, using the standard Gaussian distribution
$$
q(\mathbf{z})
=
\frac{|\mathbf{A}|^{1/2}}{(2\pi)^{M/2}}
\exp
\left\{
-\frac{1}{2}
(\mathbf{z}-\mathbf{z}_0)^{\mathrm T}
\mathbf{A}
(\mathbf{z}-\mathbf{z}_0)
\right\}
=
\mathcal{N}
(\mathbf{z}\mid \mathbf{z}_0,\mathbf{A}^{-1})
\tag{4.134}
$$
where $|\mathbf{A}|$ denotes the determinant of $\mathbf{A}$. This Gaussian distribution will be well defined provided its precision matrix, given by $\mathbf{A}$, is positive definite, which implies that the stationary point $\mathbf{z}_{0}$ must be a local maximum, not a minimum or a saddle point.

**Cons**
- Multimodal distributions: Laplace only captures one mode. 
- $A$ and $z_{0}$ depend only on the behaviour of $f$ at the single point $z_{0}$. The approximation knows nothing about the distribution's shape elsewhere, so it can badly miss global features like heavy tails or skew.
**Pros**
- The normalisation term $Z$ is not needed

## Bayesian Logistics Regression
After having Laplace approximation, we now turn to a Bayesian treatment of logistics regression. Exact Bayesian inference for logistics regression is intractable. In particular, evaluation of the posterior distribution would require normalisation of the product of a prior distribution and a likelihood function that itself comprises a product of logistics sigmoid functions, one for every data point. Here we consider the application of the Laplace approximation to the problem of Bayesian logistics regression.

### Laplace approximation
We begin with a Gaussian prior, which we write in the general form
$$
p(\mathbf{w})= \mathcal{N}(\mathbf{w}\mid \mathbf{m}_{0},\mathbf{S}_{0})
\tag{4.140}
$$
where $\mathbf{m}_0$ and $\mathbf{S}_0$ are fixed hyperparameters. The posterior distribution over $\mathbf{w}$ is given by
$$
p(\mathbf{w}\mid \mathbf{t})
\propto
p(\mathbf{w})p(\mathbf{t}\mid \mathbf{w})
\tag{4.141}
$$
where
$$
\mathbf{t} = (t_1,\ldots,t_N)^{\mathrm T}.
$$
Taking the log of both sides, and substituting for the prior distribution using (4.140), and for the likelihood function using (4.89), we obtain
$$
\begin{aligned}
\ln p(\mathbf{w}\mid \mathbf{t})
&=
-\frac{1}{2}
(\mathbf{w}-\mathbf{m}_0)^{\mathrm T}
\mathbf{S}_0^{-1}
(\mathbf{w}-\mathbf{m}_0)
\\
&\quad
+
\sum_{n=1}^{N}
\left\{
t_n \ln y_n
+
(1-t_n)\ln(1-y_n)
\right\}
+
\mathrm{const}
\end{aligned}
\tag{4.142}
$$
where
$$
y_n = \sigma(\mathbf{w}^{\mathrm T}\phi_n).
$$
To obtain a Gaussian approximation to the posterior distribution, we first maximize the posterior distribution to give the MAP (maximum posterior) solution $\mathbf{w}_{\mathrm{MAP}}$, which defines the mean of the Gaussian. The covariance is then given by the inverse of the matrix of second derivatives of the negative log likelihood, which takes the form
$$
\mathbf{S}_N
=
-\nabla\nabla \ln p(\mathbf{w}\mid \mathbf{t})
=
\mathbf{S}_0^{-1}
+
\sum_{n=1}^{N}
y_n(1-y_n)\phi_n\phi_n^{\mathrm T}.
\tag{4.143}
$$
The Gaussian approximation to the posterior distribution therefore takes the form
$$
q(\mathbf{w})
=
\mathcal{N}
(\mathbf{w}\mid \mathbf{w}_{\mathrm{MAP}},\mathbf{S}_N)
\tag{4.144}
$$
Having obtained a Gaussian approximation to the posterior distribution, there remains the task of marginalizing with respect to this distribution in order to make predictions.

According to the central limit theorem,
	Take any population with mean $\mu$ and finite variance $\sigma^2$ without any assumption on distribution. Draw $N$ independent samples $X_{1}, \dots, X_{N}$ and compute their sample mean $\bar{X}_{N}$. Then as $N \rightarrow \infty$, the rescaled quantity $\sqrt{N}(\bar{X}_N - \mu )$ converges in distribution to $\mathcal{N}(0, \sigma^2)$. 
as the number of observations grows, posteriors $p(\mathbf{w}\mid \mathcal{D})$ genuinely do become Gaussian-shaped, so the approximation improves. 


