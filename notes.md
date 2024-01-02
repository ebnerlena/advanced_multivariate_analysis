# AMA Notes

## Introduction

**Supervised Learning** (the prediction problem):

- Regression: Predicting of a quantitative response.
- Classification (or discriminant analysis): Predicting a qualitative variable
- Main interest: measure the error of predicting y with y^ in the conditional distribution Pr(y|X=x)

**Unsupervised Learning** (to learn relationships and structure from data):

- Density estimation (histogram, kernel density estimation, ...)
- Clustering (hierarchical clustering, k-means, ...)
- Dimensionality reduction (PCA, MDS, principal curves, ISOMAP, manifold learning, ...),
- Detecting communities in social networks
- Extraction of latent variables
- Main interest: To infer properties of Pr(x)

**Prediction problem**: To look for a prediction function h : X ↦→ Y such that h(X) is close to Y in some sense.
The (lack of) closeness between h(X) and Y is usually measured by a **loss function** (cost function e.g. MSE, MAE, Log Loss) L(Y , h(X)).

**Decision problem**: To find the prediction function h : X ↦→ Y that **minimizes** the expected loss.
Minimizing the loss function leads to better model performance.

**Statistical nonparametric regression estimators**: local averages (kernel regression, k nearest neighbors), local polynomial regression, spline smoothing, (generalized) additive models, CART (Classification and Regression Trees)

**Machine learning prediction models**: Neural networks, support vector, machines, ensemble meta-algorithm (random forest, XGBost, ...)

**Regression problem**: predict Y from known values of X
Most common and convenient loss function is the squared error loss: L(Y , h(X)) = (Y − h(X))²
The expected loss is known as Prediction Mean Squared Error (PMSE): PMSE(h) = E (Y − h(X))²

Parametric regression models assume that m(x) is known except for a finite number of unknown parameters.
Nonparametric regression models do not specify the form of the regression function m(x).

### k-nearest Neighbors

- Closeness is defined according to a previously chosen distance measure d (t, x), for instance, the Euclidean distance.
- Nk(t) is the neighborhood of t defined by the k closest points xi in the training sample
- m^(t) = 1/(|Nk(t)) \* SUM(yi)

**Classification Problem**: predict Y from observed values of X and use misclassifications in zero-one loss function.

## Density Estimation

- used to estimate the probability density function (PDF) of a random variable
- goal is to model the underlying distribution of the data without assuming any specific parametric form, such as normal distribution.
- Pr(a <= X <= b) = Integral of a to b f(x)dx
  - f(x)=probability/length
  - we want to estimate the value of f(x) in a nonparametric way

### Histogram

- simplest method involves dividing the range of the data into bins and counting the number of data points in each bin
- for b0 < b1 < . . . < bm define the invertval Bj = (bj-1, bj)
- compute nj = #{xi in Bj}; pj=nj/n
- over the interval Bj draw a rectangle with height fj such taht its area are equals pj
  - fj = pj = pj/(bj- b(j-1)) = estimated probability/length
- the height of each bin is proportional to the estimated density
- we want to estimate the density function f at point x
  - if x does not belong to Bj, f(x) = 0
  - if x belongs to Bj, f(x) = the height of fj = f^H(x) = fj = pj/bj-b(j-1)
- Histogram estimator: f^H(x) = SUM(pj/bj-b(j-1)/Bj(x))
- usually the intervals have the same width bj-b(j-1) = b
- f^H(x) is a true densitiy function it it non-negative and it integrates up to 1
- it is a non-continuous step function
  - it is not smooth: it is discontinuous and piece-wise constant
- very simple in calculation and interpretion
- it's appearance strongly depends on the width of intervals b
  - when b is large: histogram has high bias and low variance
  - when b is small: histogram has low bias and large variance

### Kernel Density Estimator

- nonparametric estimator that outperform the histogram
- smooths the data by placing a kernel (smooth, symmetric function) at each data point and summing up these kernels to estimate the density
- the bandwidth of the kernel determines the amount of smoothing
- 1. **localization**: moving histogram
  - the histogram estimated better the density function at the center of each interval than at its boundaries
  - we force x to be center of one of the histogram intervals when we want to estimate the density for x
  - so move the density around x
- 2. **smoothing**
  - using a kernel function K(u) for estimating the density to obtain smoothness, contiousness, unimodel and symmetric around 0
  - h is known as smoothing parameter = bandwidth
    - if small: only the observations xi that are close to x are relevant in the estimation, large variance -> very differend from sample to sample, in average captues the unknown density function -> small bias
    - if large: also far away observations from xi are considered in estimation, small variance but largee bias, stable from sample to sample
- f^K(x)= 1/n _ (SUM(1/h _ K ((x-xi)/h)))
- the kernel density estimator spreads the weight 1/n of each observation in its neighborhood in a continuous way
- the bandwidth is choosen by maximum likelihood cross validation based on LOO
  - for a given h, the likelihood of observation xi is evaluated using the density estimator computed from the other elements in the sample
  - we look for the maximum value hLVC(h) = Product of f^h(-i) (xi)
  - it is enough to perform the estimation just once using all observations
  - in practice it is advised to use the logarithm of the LOO Likelihood

**Multivariate Density Estimation**

- estimate the joint probability density function (PDF) of multiple random variables
- placing a kernel at each data point and summing up these kernels to estimate the joint density
- H is a non-singular bandwidth matrix = a diagonal matrix H = Diag(h1,...,hp)
- fˆ(x) is a mixture of n densities, each of them centered at one of the observation

**R Software**:

- Functions `hist`, `density`. Bandwidth choice: `bw.nrd`, `bw.ucv`, `bw.bcv`, `bw.SJ`
- Package `KernSmooth` function `bkde` (for density estimation) and `dpik` (for bandwidth choice)
- Package `sm` functions `sm.density` (for density estimation) and `h.select` (for bandwidth choice)
- Package `ks` function `kde` (for density estimation) and several functions for bandwidth choce

### Gaussian Mixture Models GMM

- assumes that the data is generated from a mixture of several Gaussian distributions
- estimates the parameters (means, covariances, and weights) of these Gaussians to model the overall density

---

## Clustering

- the goal is to group a collection of objects into subsets or clusters C1,...Ck
- in such way that those within each cluster are more closely related to each other than the objects assigned to different clusters
- the right number of clusters is not known in advance
- objects are clustered by their set of attributes, features or variables by its distance or similarity

**Distances / Similarities**

- **distance d**: Ω × Ω → R is symmetric, non-negativ, triangular inequality
- **similarity s**: Ω × Ω → R is symmetric, non-negativ, an s(P,Q) is an increasing function of the proximity of P and Q
  - kernel functions can be used as similarity measure (e.g. like in SVM)
- if d(P,Q) is a distance -> s(P,Q) = 1/(1 + d(P,Q)) is the similarity
- if s(P,Q) is a kernel similarity function -> d(P,Q) = SQRT(2(1-s(P,Q))) is the distance

- distances can be measured with
  - euclidean distance with L2-norm (numeric attributes)
  - mikowski distance (numeric attributes)
  - cross table (for binary attributes)
  - mismatch distance (for categorical attributes)
  - normalized distance with ranks rx and ry (for ordinal attributes) |rx-ry|/m
  - gower's distance (for mixed variables) using any distance measure for each variable and aggregating them

### Hierachical Clustering

- provides a list of n consecutive partitions of the set of the n observed objects
  - first partition contains only one cluster
  - second partition contains two clusters
  - ....
- partitions Cg and that with Cg+1 contain (g-1) identical clusters
- the list of partitions are usually represented graphically as dendrogram (rooted binary tree, where root node represents the entire data set)
- n levels in the hierachy

Strategies

- **agglomerative** (bottom-up)
  - start at bottom and at each level recursively merge a selected pair of clusters intro a single cluster
  - this produces a grouping at the next higher level with one less cluster
  - the pair chosen for merging consist of the two groups with the smallest between-group dissimilarity (using distance matrix D)
- **divisive** (top-down)
  - start at top and at each level recursively split one of the existing clusters at that level into two new clusters
  - the split is chosen to produce two new groups with the largest between group dissimilarity

Cluster Linkage

- **single linkage** (nearest neighbor)
  - d(UV),W = min{dU,W , dV ,W }
  - join clusters by the shortest link between them
  - can discover non-ellipsoidal clusters
- **complete linkage** (furthest neighbor)
  - d(UV),W = max{dU,W , dV ,W } =
  - tends to produce compact clusters with small diameters
- **average linkage**
  - compromise between single and complete linkage
  - d(UV),W = SUMi SUMj dij/(#(UV) \* #(W))

**Ward's Hierachical Clustering**

- based on minimizing the loss of information from joining two groups
- Loss of information = Increase in the sum of square distances
- for cluster U the sum of square distances is SSD = SUM(dij²)
- SUM(SSD Uj) is the total sum of square distances of the partition Uj
- the union of every possible pair of clusters is considered and the two clusters whose combination results in the smallest increase in SSD are joined

### K-means

- a non-hierachical clustering method
- can be used when the infromation observed at each object is a p-dimensional numerical vector
  - because it requires to compute the average of the observations belonging to each cluster
- K-medoids is a generalization of K-means that only needs a distance matrix between objects (no averages must be computed)
  - more robust then k-means
- the number of final clusers k is known in advance

**Algorithmn**:

- divide data into U1,...Uk random partition of K clusters
- for the given clusters K find the mean of each cluster that will be the current estimates of the clusters center
- given a current set of clusters centers {m1,...mk} miminize the total sum of distances by assigning each observation i to the closest cluster center
  - if argmin k(i) = d(xi,mk) assign i to cluster k(i)
- iterate until the assignments do not change

### Clustering Quality

- how many clusters are in a dataset?
- how to cut the dendrogram representing a hierachical clustering?
- how to determine the number of K lcustes when applying k-means or k-Medoids?

**Determining the number of clusters**:

- visual inspection of the dendrogram
- **gap statistic**
  - goodness-of-clutering measure
  - compute clusters sum of squared distnaces for a partition in K clusters
  - compares performance of clustering algorithm on the actual data to its performance on a reference dataset with no inherent clustering structure
    - generate a reference dataset by randomly permuting vlaues of each variable independently
  - compute difference betweent the log of the reference and actual clustered data
  - repeat for different numbers of clusters and choose the number of clusters where the gap statistics is maximized -> smallest K producing a gap within one sd
- **silhoutte**
  - measures how well-defined the clusters are in a given partition
  - for each data point calculate the silhoutte score [-1,1]
    - high score indidcates that data point is well matched to its own cluster and poorly to neighboring clusters, 0 mean that observatio lies between 2 clusters, -1 means probaly placed in wrong cluster
  - compute average silhoutte score for all data points which gives an indication of how-well-separated the clusters are
  - repeat for different numbers of clusters
  - choose the number k of clusters that maximizes the score
- **calinski and harabasz index**
  - measures the ratio between the variance of the between-clsuter dispersion and the variance of the within-cluster dispersion
  - aims to classify the compactness of clusters and the seperation between them
  - 1. Compute the Between-Cluster Dispersion: across cluster, reflects how much cluster centers differ from each other
  - 2. Compute the Within-Cluster Dispersion: wihtin each cluster, reflects the compactness of the clusters
  - 3. compute Calinski-Harabasz Index: CH = Between-Cluster Dispersion/Within-Cluster Dispersion x (n-k)/(k-1)
  - repeat for different numbers of clusters
  - choose number of k that maximizes the score

### Mixture Models for Density Estimation and Clustering

- mixture of several underlying subpopulations or components, each characterized by a probability distribution
- the assumption that the true density function f(x) follows a mixture model is useful for density estiamtion when p is large
- parametric mixture model: f(x) = SUM(αj fj(x; θj))
- e.g gaussian mixture model is density funciton of a multivaraite normal with mean vector µj and variance matrix Σj
  - the underlying components are assumed to be Gaussian (normal) distributions
  - is a parametric model that assumes that the observed data is generated from a mixture of several Gaussian distributions
  - each component in the mixture represents a cluster in the data, and the overall distribution is a weighted sum of these components
- Clustering based on parametric mixture models is called **model-based clustering**
- fitting of a GMM to data is often done using the **Expectation-Maximization (EM) algorithm**
  - iteratively estimates the parameters of the model by alternating between an "expectation" step (E-step) and a "maximization" step (M-step) until convergence
- determining the number of compontents is a critical task; crterions are AIC, BIC

### Dbscan

- density based spatial cluserting with noise
- groups data points together based on their density in a feature space
- can discover clustrs of arbitrary shapes and sizes (k-means assumes spherical shape and are equally sized)
- **clusters** = connected areas of the sample space with high density of observations
- **noise** = observed points not belonging to any cluster
- ε-neighborhood of xi = Nε(xi) = {xj ∈ D : d(xi, xj) ≤ ε}
- the density of ovservations around xi is estimaed by the number of observations at Nε(xi): #Nε(xi)
- DBSCAN parameters:
  - small distance ε > 0
  - minimum number of points minPts
- **Core points**: a point xi if #Nε(xi) >= minPts (if estimated density of xj is over a certain threshold)
- **Border points:** a non-core point xj if there is at least a core point xi such that d(xi,xj) <= ε
- **Outlier points**: points that are neither core points nor border points are said to be outliers
- **Density connected**: two points are density connected if there is a path of core points x1,...xm such that d (xj, x(j+1)) <= ε
- Clusters are subsets of density-connected points (all points in a cluster are density connected)
  - if a point xi is density connected with another point xj, then both belong to the same cluster - only core and border points belong to clusters
- outliers do not belong to any cluster
- the number of clusters must not be specified in advance
- the core points in a cluster are a subset of observed points belonging to the same connected component of the level set of the estimated density function
- dbcan is robust against outliers

**Algorithm**:

1. select hyper-parameters ε and minPts
2. mark each observed points as either core, border or outlier
3. remove outliers from dataset
4. set j = 1
5. select a random core point and define the cluster Cj = {x1j}
6. add to Cj all the core points that are density connected to x1j
7. add to Cj all the border points that re density connected to x1j
8. remove cj from the dataset
9. repeat form step 5 until there are no core points left

### R:

- `hclust`, `plot.hclust`, `cutree` for hierarchical methods
- `kmeans` for K-means
- `pam` (package cluster) for K-medoids
- Package `ClusterR` includes K-Means, K-Medoids and Gaussian Mixture Models (GMM)
- Gap statistic. See the function `clusGap` of the R package `cluster`
- Silhouette. See the function `silhouette` of the R package `cluster`
- Calinski-Harabasz index. See the function `cluster.stats` of the R package `fpc`
- Package `ClusterR` includes functions that determine the optimal number of clusters in K-Means, K-Medoids and GMM.
- Function `GMM` fits Gaussian Mixture Models using the EM algorithm
  - includes the function `Optimal_Clusters_GMM `to determine the optimal number of clusters in GMM
- `ClusterR` also performs other clustering methods as K-Means and K-Medoids
- Package `mclust`: package for model-based clustering, classification and density estimaiont baed on normal mixture modeling
  - function `Mclust`: model based clsutering based on a parameterized finite GMM, estimated with EM and selected best model with BIC
- Package `fpc`:
  - `mergenormals`: cluserings by merging components of a Gaussian mixture
  - `cluster.stats`: computes several cluster validity statistics fomr a clustering and a dissimilarity matrix
  - `dbscan`: Computes DBSCAN density based clustering
- Package `dbscan`, function `dbscan`: Fast reimplementation of the DBSCAN.

---

## Dimensionality Reduction

---

Local regression techniques are particularly useful when dealing with complex, non-linear relationships that may vary across different parts of the dataset.

## Non Parametric Regression

- the goal is to model the relationship between the predictors and the response variable without making strong assumptions about the functional form of that relationship
- allow the data to dictate the shape of the relationship

## Local Polynomial Regression

- the best prediction of the dependent variable Y given that predicting variable X takes the known value x it the conditional expectation of Y given that X = x -> m(x) = E(Y|X=x)
- simple linear regression: y = β0 + β1x + ε
- where betas are unknown parameters
- nonparametric regression model: yi = m(xi) + εi
- fitting a nonparametric regression model = to provide an estimator m(t) of m(t) to give an estimator σ²2 of the residual variance σ²

**Local Linear Regression**

- can capture non-linear relationships
- fits a linear model to a subset of the data, emphasizing a specific region around a given point
- often used when the relationship between variables is expected to vary across different parts of the data
- fit a linear model within a local neighborhood of each data point, giving more weight to points that are closer to the point of interest
- improve local linear regression with localization (centering each interval at t) and smoothing (assigning weights to each observation being a decresing function of distance |t-xi|)
- weights are assigned by a kernel function K
- h as smoothing parameter = bandwidth (controlling size of local neighboorhood and smoothness)
  - if small only the closest observations to t have a relevant weight
  - ig large allows distant observatoins to be taken into account for estimating m(t)
- this solves the weighted least squares problem
  - minimizing the sum of squared differences between observed and predicted values, with weights assigned to each data point based on its proximity to the point of interest
- a kernel function is used to assign weights to data points
- for each data point, a linear model is fit using a weighted least squares approach
- the local linear fits adapt to changes in the underlying relationship, providing a smooth transition

**Local poylnomial fitting**:

- low-degree polynomial provides a smooth fit, while a higher-degree polynomial can capture more complex, non-linear relationships
- using multiple unknown parameters β1....βq
- weighted polynomial regression: the estimated coefficients depend on t, the point for which the regression function is being estimated
- -> min SUM(wi(yi − (β0 + β1(xi − t) +...+ βq(xi − t)^q))²)
- the estimate for m(t) is the locally fitted polynomial evaluated at x=t -> mˆq(t) = βˆ0(t)
- when the degree of the locally fitted polynomial is q=0 (constant) the resulting nonparametric estimator of m(t) is a kernel estimator with moving weighted average
- local polynomial regression estimator is a linear smoother (because for a fix t m^q(t) is a linear function)
- polynomial degree 1 equal local linear regression
- Nadaraya-Watson estimator is used for estimating the conditional expectation when using q=0 -> knows as kernel regression/kernel smoothing estimator
  - assigning weights to each observation based on its distance to the point x0 and then using these weighted observations to estimate the conditional expectation

**Linear Smoothers**:

- smoothing data by fitting a linear model to data to reduce noise, emphasize trend and capture underlying patterns in data
- e.g. moving average smoother, kernel smoother, splines, local linear regression
- a nonparametric regression estimator is a linear smoother when for any fix t, m^(t) is a linear function
  - m^(t) = SUM(w(t,xi)yi)
- are particular cases of linear estimators of the regression function
- matrix form: Y^=SY where S is the smoothing matrix
  - the sum of diagonal of S is the effective number of parameters v
- for any linear non parametric estimator (e.g. local polynomial regression) v is a decreasing function of smoothing parameter h
  - small values of h correspond to large numbers v of effective parameters (highly complex and very flexible)
  - large values of h correpsond to small number v of effective parameters (low complexity and flexibility)
  - we can compare the degree of smoothing of two estimators by just comparing their effective numbers of parameters
- estimaor of σ² = 1/n-v \* SSR
  - SSR = SUM(yi - m^(xi)²)
- variance explained = σ²/var(y)

**Kernel Functions**:

- a weight function that assigns weights to data points based on their distance from a specific point of interest
- the purpose is to give higher weight to nearby data points and lower weight to more distant points when fitting a local polynomial model
- these kernel functions are typically symmetrix and centered at zero
- bandwidth parameter h determines the widht of the neighboorhood considered
  - larger h -> wider neighboordhood, smoother but less sensitive
  - smaller h -> narrower neighboorhood, capturing more details but introducing more variability
- different shapes:
  - Gaussian Kernel (Normal distributed) -> 1 variance
  - Epanechnikov Kernel (runder Bogen) -> 1/5 variance
  - Biweight Kernel 1/7 variance
  - Triweight Kernel -> 1/9 variance
  - Triangular Kernel (Dreieck) -> 1/6 variance

#### Bias-Variance Trade-Off

- local behavior: tatistical properties of nonparametric estimator as estimator of unknonw value m(t) for a fixed value t
- global properties: when estimator considering all t in an interval
- distance from m and m^ is measured by Integrated Mean Squared Error
- AMSE for local linear estimator
- the optimal value of AMSE(h) represents a compromise between bias and variance (increases with h), because the AMSE is the squared bias + variance (decreases with h)

#### Bandwidth Choice

- choice of smoothing parameter h is crucial for appearnce and properties of regression estimator
- it controls the bias-variance trade-off
- **estimation**:
  - small h: high variance (applied to different smaple from the same model gives very different results), small bias (average of the estimarots obtained for different sample si ~ the true regression function)
  - large h: small variance, high bias
  - criterion: IMSE, MISE -> depend on m(t)
- **prediction of new observations**:
  - h controls the balance between fitting the observed data well and the ability to predict future observations
  - small h: great flexibility, high prediction errors, overfitting as it approaches all observed data
  - large h: underfitting, errors in observed sample and prediction errors will be high
  - criterion: PMSE (expected squared error made when predicting) -> depends on m(t)
- feasible criterion = RSS, that is optimistically biased
- bandwidth choice with
  - minimizing average squared prediction error in validation set (prediction error PMSE)
  - Leave one out cross validation
    - when sampling size does not allow us to set a validation set aside
    - an approximately unbiased estimator of PMSE(h) but has high variance
    - remove the observation (xi,yi) from the sample and fit nonparametric regression using other (n-1) data
    - use resulting estimarot m^hi(xi) to predict yi and compute PMSE(h) for all bandwidth candidates = 1/n SUM(yi - m^hi(xi))²
    - variance can be reduced with k-fold cross validation
    - does require a validation set
  - k-fold cross validation
    - randomly divide sample in k subset and remove each of subset once and estimate with rest of subsets and compute PMSE
    - has lower variance then looc but larger bias
    - use 5-fold or 10-fold cv
    - PMSE in cv can be calcualted more efficiently avoiding the cost of fitting n different models -> PMSE cv (h) = 1/n SUM((yi-yi^)/(1-sii))
  - generalized cross validation (defined only for linear estimator of regression function)
    - replacing values sii comding from the diagonal of smoothing Matrix S by their average value in PMSE calculation
      - v = SUM sii = effective number of parameters
      - σ²ε = 1/n-v SUM (yi-y^i)²
      - PMSE gcv(h) = n \* σ²ε/n-v
  - Plug-in: specific bandwith selector for local linear regression

#### Choosing the degree q of the local polynomial

- less important than bandwith choice
- the larger q the better are the asymptotic properties (in bias)
- in practice it is recommended to use q=r+1
  - r is the order of the derivative of m(t) that is estimated
- it is prefered to use the odd degree q = 2k +1 for estiamting
  - odd degrees are able to automatically adapt to the boundary of the explanatory variable
- decide if it is worth to fit a local cubic model q=3 instead of local linear model q=1
  - bias is high for t in intervals where the funciton m(t) has high curvature
  - therefore it would be better to use q=3 instead of q=1 in that case

### Generalized Non Parametric Regression

- non parametric versino of generalized linear model GLM
- different types of response var Y e..g binary, count variable, non-negative
- the conditional distribution of (Y|X=x) is in a parametric (exponential) family
  - one of the parameters is a smooth function of x

**Logistic Regression**:

- regression with binary response
- the conditional distribution of Y given X=x is a Bernoulli(p(x))
- p(x) is known as logistic function
- log(p/1-p) is known as logit function -> in the logistic regression function it is the link function because it links the conditional expectation P(x) with the linear term β0 + β1x
- the estimation of parameters β0 and β1 is done by maximizing the log-likelihood function with IRWLS (iteratively re-weighted least squares)
  - iterate until convergence

**Generalized nonparametric regression model**

- (X,Y) has a joint distribution of (Y |X = x) ∼ f (y; m(x), ψ)
  - where m(x) = E(Y|X=x) is a smooth function of x
  - ψ are other parameters not depending on x e.g. variance
- an invertible link function g() exists θ(x) = g(m(x)), m(x) = g^−1(θ(x))

**Estimation by maximum local likelihood**

- we fit the logistic model by maximizing the local likelihood
- the contribution to the log-likelihood function of each observation is yi\*log((pit)/1-pit) + log(1-pit)
- the closer xi is to t, the better is the approximation
- adding up all the data contributions weighted -> the local log-likelihood function is obtained
  - lt() = SUM (wit * (yi*log((pit)/1-pit) + log(1-pit)))
- the logisticc model is estimated by a weighted version of the IRWLS algorithm
  - multiplying at each iteration the usual weights pis(1-pis) by the kernel weights wit
- once the parameters βˆt0 and βˆt1 are obtained, the function θ(t) and p(t) can be estimated

**Bandwidth Choice**

- minimizing in h the probability of misclassification of a new observation
  - using leave-one-out cross validation
- maximizing in h the expected log-likelihood of a new observation
  - using leave-one-out cross validation
- using cross-validation

### Spline Smoothing

**Penalized least squares nonparametric regression**:

- non parametric regression model yi = m(xi) + εi
- expressing the estimation problem as an optimizatioun problem
- solving the penalized least squares problem
  - adding a penalizaiont by the lack of smoothness to the Least Squares problem
  - unique solution: cubic spline with knots at x1...xn (the observed values of the explanatory variable)
- involves fitting a piecewise-defined polynomial (spline)
- the spline is usually continuous across segments
- the fitted splines consists of multiple polynomial segments connected at specific points called knots
- selecting a appropriate number and positions of knots or tuning the smoothing parameter is crucial
- flexibility is controlled by the choice of knots

**Natural Cubic Splines**

- using piecewise q=3 polynomials that are continuous and have first and second continuous derivatives at the knots
- have additional constriant at the boundaries to ensure smoothness
  - setting the second derivatives to zero at the endpoints
- Natural cubic splines are appropriate for achieving smoothness, especially at the endpoints
- periodic spline: form a closed loop (smoothly connects the last and first point) - repeats pattern over a specific interval
  - setting the first and second dervivatives to be qual at the first and last knot
  - helpful for timeseries data
- use cubic polynomials between knots

**Smoothing Splines**:

- cubic splines that minimalize the penalized sum of squared differences
- the degree of smoothness is controlled by a smoothing parameter
- the spline estimator of m(x) is a linear smoother
  - smoothing parameter choice with loocv/gcv
  - efficient computation of loocv
  - effective number of parameters
  - estimation of the residual variance

**B-Splines**

- = basis splines
- functions that form a basis for a space of piecewise-defined splines
- basis for S[p; a = t0,t1, . . . ,tk ,tk+1 = b]
  - degree p and knots t1...tk
- basis of B-Splines of order 1 (degree 0) Bj,1 = I[τj,τj+1], j = 1, . . . , k + 2M − 1
  - 2M are auxilary knots
  - m=M=4 is the basis of cubic splines

**Regression Splines**

- in practice it is not necessary to look for those having knots at every observed xi, as computational cost for large values of n is high
- it is enough to take a sufficiently large number of k knots
- knots can be evenly spaced or they can be the quantiles
- tunning parameters, λ and k play the role of smoothing parameters
  - λ = 0, the number of knots k acts as smoothing parameter
  - if k is fixed and large, then λ = 0, t is the only smoothing parameter (common option)

**Generlaized nonparametric regression with splines**

- use Penalized Iteratively Re-Weighted Least Squares (P-IRWLS)
  - at each step of the IRWLS algorithm a glm model is fit and the linear fit is replaced by spline smoothing
- alternative: To fit a GLM using a cubic B-spline basis matrix as regressors
  - the number of knots k controls the amount of smoothing

### Generalized Additive Models GAM

- useful when dealing with complex relationships and patterns in the data

**Local polynomials for multiple nonparametric regression**

- multiple nonparametric regression yi = m(xi1, . . . , xip) + εi
- the local polynomial estimator of the regression function m(x) needs
  - the weights wi for each observation
    - measure distances between t and xi in p-dimensional space, where ci closer to t should have a greater weight
  - specify which explanatory variables are inclued at each local linear regression model - the estimate of m(t) will be the intercept of the local polynomial fitted around point t

**Tensor Product Splines**

- a type of spline basis that is formed by taking the tensor product of one-dimensional spline bases
- these bases are used to represent smooth functions in two or more dimensions
- the spline basis is constructed by taking the tensor (outer) product of the one-dimensional spline bases
- a tensor product basis of functions for functions depending on both x and z
- main drawback: exponential growth in basis functions as p increases

**Curse of dimensionality**:
= In high dimensional spaces the neighborhood of any point contains virtually no observational data

- it is recommended not to go beyond 3 or 4 dimensions
- the higher the dimension p of explanatory variable, the lower the precision with which the regression functoin is estimated
- additive models and projection pursuit are proposals to overcome the curse

**Additive Models**:

- nonparametric regression models that are less flexible than the multiple nonparametric regression model
- are good when number of explanatory variables is high
- are able to overcome the curse of dimensionality
- yi = α + SUM(gj(xij) + εi)
- functions gj must be estimated nonparametricaly because no parametric model is specified for them
- the nonparametric univariate functions gj are combined additively to produce the nonparametric p-dimensional regression function
- halfway between the multiple linear regression model (which combines linear transformations of the explanatory variables) and the multiple non parametric regression model
- E(yi) = α, because E(εi) = 0 and E(gj(Xj)) = 0
- the additive model is estimated with **backfitting**
  - by estimating α by α^=(1/n)SUM(yi) until convergence
- the GAM model is estimated as a penalized multiple linear regression model with 1 + SUM(Hj) parameter
- smoothing parameters λ1, . . . , λp are chosen by LOOCV or GCV

**Generlized Additive Models GAM**

- with restriction that functions gj are linear in α + SUM(gj(xj))
- halfway between the generlized nonparametric multiple regression model and the Generalized linear model
- estimation of a GAM model combins the methods to fit additive models with the IRWLS algorithm (used to maximize the likelihood in GLM)
- in IRWLS, each multiple linear regression fit is replaced by fitting of a weighted additive model
  - using backfitting or penalized multiple linear regression

**Semiparametric Models**

- if the response variable is linear the GAM can be reformulated as gj(xj)=βjxj
- nonparametrically estimating the combined effect of two (or more) explanatory variables e.g. replacing gj(xj) + gh(xh) by gjh(xj, xh)
- estimating the effect of a variable xj differently at each of the lcasses determined by another categorical variable xh (can be done linearly or nonparametrically)
- can be fitted using the `gam` R function of the `mgcv` package

### Local Poison Regression

- is a form of nonparametric regression

Local Poisson regression refers to a statistical modeling technique that extends traditional Poisson regression by allowing for local variations in the relationship between the predictor variables and the response variable.
In other words, instead of assuming a constant relationship across the entire dataset, local Poisson regression models the relationship as varying across different regions or subsets of the data.

#### Poisson Regression:

Poisson regression is commonly used when the response variable represents counts or event occurrences in a fixed period of time or space.
It assumes that the mean of the response variable is a function of predictor variables, and the response variable follows a Poisson distribution.

#### Local Poisson Regression:

Local Poisson regression allows for local adaptation of the Poisson regression model. This means that the relationship between predictors and the response can vary across different parts of the dataset.
Local regression models are particularly useful when there is evidence that the relationship between variables is not constant but changes in different regions of the predictor space.
This local adaptation is often achieved through the use of kernel functions or other smoothing techniques. The model estimates parameters locally around each data point.

#### Applications:

Local Poisson regression can be applied in various fields, including epidemiology, ecology, and other areas where count data are prevalent.
It is often used when there is a suspicion that the relationship between predictors and counts may vary across different conditions or subsets of the data.

#### Implementation:

The implementation of local Poisson regression may involve specialized statistical software or packages that support local regression techniques. In R, for example, the loess function can be used for local regression.

---

## Interpretable Machine Learning

- Desirable properties for predictive models: transparency, interpretability, explainability
- predictive capacity vs. interpretability
  - Tradidiontal Statistics: **transparent models** (LM, GLM, GAM, CART, knn, Bayesian models)
  - ML models: **black boxes** -> low interpretability

**Global interpretability**: Measures of variable importance or relevance.

Information about the global performance refers to determining which is the role of each explanatory variable in the prediction process over the whole support of the explanatory variables.

**Local interpretability**: Why the prediction model does a particular prediction for a given individual?

The goal of understanding local performance is to provide a meaningful explanation of why the algorithm returns a certain prediction, given a particular combination of the predicting variables values.

Non-transparent models can be divided into two subgroups:

- **model-specific** methods:
  - require full access to the model structure
  - Tree ensembles, RF, Boosted Methods, NN, DL, CNN
- **model-agnostic** methods:
  - no need to know internal structure of the prediction model
  - only requirement: the ability to evaluate the prediction model repeated teims on data from the training or test set or perturbations of them
  - can be applied to any predictive model, even to those having model-specific methods or those that are transparent models
  - global vs. local measures

### Interpretability for Specific Methods

- Interpretability methods developed for a particular prediction method
- Require full acess to model structure
- Difficult to compare between differend prediction models

#### Random Forest

- Random forests are combinations of more simple models: classification and regression trees (CART)
- CART are usually considered transparent models because the prediction rules they encode are easily understood by non-expert users
- At each split in the tree, the improvement in the split-criterion is the importance measure attributed to the splitting variable.
- In random forests, this importance measure is accumulated over all the trees in the forest separately for each variable.
- Tree-based methods divide the feature space into a set of regions, and then fit a simple model (like a constant) at each one
- the algorithm needs to automatically decide on the splitting variables and split points

**Regression Trees**:

- p inputs and a quantitative response
- we have a partition into M regions R1, . . . , RM
- we model the response as a constant cm in each region f(x) = SUM(cm | Rm(x))
- node impurity criterion: minimization of the sum of squares SUM(yi − f (xi))² the best value for cm is the average of yi in region Rm
- greedy strategy as finding the best binary partition is computationally infeasible

  - recursive top-down approach
  - select a variable and split point (to minimze sum of squares in region)
  - choose best split based on impurity measure
  - create child nodes and repeat recursively until stopping criterion is met
  - assign predictions to each leaf node (mean of target values in leaf node)

- large trees might overfit the data, while small trees might not capture the important structure
- grow a large tree and stop splitting only when some minimum node size is reached
- this large tree is pruned using **cost complexity pruning**
  - Pruning helps simplify the tree by removing branches that do not contribute significantly to the overall predictive performance
  - assign a cost to each subtree based on a trade-off between the accuracy of the subtree on the training data and the complexity (size) of the subtree
  - the cost-complexity parameter is a non-negative value, and as it increases, the penalty for adding more nodes to the tree increases

**Classification Trees**:

- target is a classification outcome taking values 1, . . . , K
- we classify the observations in node m to class k (m) = arg maxk pˆmk , the majority class in node m
- node impurity is measure by misclassification error, gini index or cross-entropy or deviance
  - for growing the tree is gini index or cross-entropy
  - for cost-complexity pruning typically misclassification rate is used

Trees have a high variance.
Small changes in data result in a very different series of splits, making interpretation precarious.
The effect of an error in the top split is propagated down to all of the splits below it.
Bagging (Bootstrap Aggregating) improve stability and accuracy of trees by creating multiple independent models, each trained on different bootstrap sample of the original data and taking average predictions of all models -> Random Forest

**Random Forest**:

- a large amount of random trees is generated and then they are averaged.
- reduces variance without increment in bias
- take bootstrap sample, choosing with replacement n random elements from the original dataset
  - with replacement: each element can be selected more than once as it remains in the pool of available elements for subsequent draws
- several real data appear at least once in the bootstrap sample (~2/3)
- other (~1/3) do not belong to bootstrap sample -> out-of-bag sample OOB
- compared to bagging it reduced correlation between trees in the tree-growing process through random selection of the input variables
- use OOB to predict random forest on those observation and evaluate OOB error by avering all of them -> almost identical to n-fold-cross-validation
- once the OOB error stabilzes, the training can be terminated -> random forest can be fitted with cross-validation being performed along the way

**Impurity Measures**:

- At each split in the tree, the improvement in the split-criterion is attributed to the splitting variable as a partial measure of its importance
- The importance measure of a variable is the sum of the partial measures of importance corresponding to all splits defined by this variable.
- In Random Forests, this importance measure is accumulated over all the trees in the forest separately for each variable
- **Out-of-Bag Variable Importance**:
  - randomly permuting the values of each predictor in a test sample to measure the decrease in accuracy
  - when b-th tree is gown the OOB samples are passed down and the prediction accuracy is recorded
  - then the values for the j-th variable are randomly permuted in the OOB samples and accuracy is computed again
  - the decrease of accuracy as a result of this permuting is averaged over all trees and used as measure of variable j in the random forest

#### Neural Networks

- inspired by the human brain
- try to mimic with mathematical models the properties observed in the biological neural systems
- we only deal with one-hidden-layer neural networks
- a one-hidden-layer neural network is a non-linear parametric regression model represented by a directed graph
- at each node N the inputs are additively combined
- then they are transformed by an activation function sigma
- a useful tool for interpretability in NN is to look at the derivatives of the prediction function
  - activation maximization: searching for the input pattern that produces a maximum model response for a quantity of interest
  - indicates which characteristics in the data are mainly taken into account by the model
- for explanation in NN
  - sensitivity analysis: goal is to identify the input feature along which the largest local variation is produced around a given data point x, e.g. compute relevance score at x for each feature h
  - simple Taylor decomposition: the NN function is approach at a given data points x by the first order Taylor expansion, which is then interpreted as any linear estimator providing an explanation of how the NN function varies around x
  - Taylor decomposition decomposes a function into an infinite sum of terms, each of which is derived from the function's derivatives at a specific point
  - provides an approximation of the function arount the point x=a

---

**Model-agnostic interpretability methods**

- only require the evaluation of the fitted prediction model on the training set, on the test set, or on perturbations of them
- to interpret the prediction model equals to interpret te prediction function f
- any procedure that allow exploring a generic function g could be used for interpreting a prediction fucntion f

**Variable Relevance**:

- = variable importance
- quadratic loss function with risk PMSE(f) = E((Y- f(X,Z))²) measuring the cost associated with predicting Y by f(X,Z)
- the problem of measuring the effect of the single variable Z on the prediction function f when predicting Y by f(X,Z)
- assumption that training sample of size n1 and test sample of size n2 are available

### Global Methods

##### LOCO

1. fit the model icluding both X and Z
2. fit the model including only X (leaving out Z)
3. relevance of Z by loco: the relative decrease in prediction accuracy in the test sample when Z is omitted from the model
4. (rank variables according their impact on model performance)

- used in multiple linear regression
- model must be fitted twice (multiple times if you want to check for all variables)
- use quadratic loss for

```{r}
library(mgcv)

# Generate example data
set.seed(123)
data <- data.frame(
  x1 = rnorm(100),
  x2 = rnorm(100),
  y = 2 * x1 + 3 * x2 + rnorm(100)
)

# Train the additive model
model <- gam(y ~ s(x1) + s(x2), data = data)

# Leave-One-Covariate-Out
for (covariate in colnames(data[, -ncol(data)])) {
  model_without_covariate <- gam(y ~ . - get(covariate), data = data)
  # Evaluate and compare model performance
  # ...
}
```

##### Random Permutations

1. train the predictive model on original dataset using all original explanatory variables
2. evaluate model on test set
3. select the variable on interest and randomly shuffle the vlaues of the selected variable Z in the testset (breaks any relationship between the selected variable and the target variable)
4. reevaluate the model on permuted testset
5. importance score of variable is calculated as difference between the originial perofrmance and the permuted variable performance (a large drop indicates higher importance)

- common in tree-based models
- measure the impact of randomly shuffling the values of a particular variable on the model's performance
- model is trained only once
- relevance measure E( (f (X, Z) − f (X, Z′)²))
- random permuations are just considering variance of Z
- cannot detect differences in relevance of Z if X and Z are independet or strongly correlated
- replacing Z with Z' results in a reduced version of f equivalent to use the mean E(Z) instead of Z
- risk of extrapolation: when X and Z are strongly correlated using a replaced version f(X,Z') the support could be much larger than the support of (X,Z)

- risk of extrapolation: when X and Z are strongly correlated using a replaced version f(X,Z') the support could be much larger than the support of (X,Z)

**extrapolation** involves making predictions for input values that extend beyond the range of the training data, assumes continuity

**Concept of permutations**:
To replace the values of Z in the test set by “perturbed” values of them, which are independent of the response variable Y , given the other explanatory variables X

<details>
<summary>CODE</summary>

```{r}
# Load required libraries
library(randomForest)
library(caret)

# Load the Iris dataset

data(iris)

# Create a binary target variable for classification

iris$Species_binary <- ifelse(iris$Species == "setosa", 1, 0)

# Split the data into training and testing sets

set.seed(123)
train_indices <- createDataPartition(iris$Species_binary, p = 0.8, list = FALSE)
train_data <- iris[train_indices, ]
test_data <- iris[-train_indices, ]

# Train a random forest classifier

rf_model <- randomForest(Species_binary ~ ., data = train_data, ntree = 100)

# Evaluate the original model

original_predictions <- predict(rf_model, newdata = test_data)
original_accuracy <- confusionMatrix(original_predictions, test_data$Species_binary)$overall["Accuracy"]

# Permutation importance calculation

permuted_accuracies <- numeric(length = ncol(iris) - 2) # Exclude target and binary Species variables

for (i in 3:ncol(iris)) { # Start from the third column (features only)
test_data_permuted <- test_data
test_data_permuted[, i] <- sample(test_data_permuted[, i])

# Make predictions on the permuted dataset

permuted_predictions <- predict(rf_model, newdata = test_data_permuted)

# Calculate accuracy on the permuted dataset

permuted_accuracy <- confusionMatrix(permuted_predictions, test_data$Species_binary)$overall["Accuracy"]
permuted_accuracies[i - 2] <- permuted_accuracy
}

# Calculate importance scores

importance_scores <- original_accuracy - permuted_accuracies

# Rank variables based on importance scores

variable_importance <- data.frame(Variable = names(importance_scores), Importance = importance_scores)
variable_importance <- variable_importance[order(-variable_importance$Importance), ]

# Display the variable importance

print("Variable Importance:")
print(variable_importance)

```

<details>

##### Ghost Variables

1. fit model with training sample and all original explanatory variables
2. evaluate model with test sample
3. define ghost variable for Z as Zx = E(Z|X) with doing the last estimation on test sample
4. evaluate accuracy of the ghost variable model with test sample
5. measure the relative decrease in prediction accuracy

- the concept of replacing Z by its conditional expectation given X provides a nuanced understanding of the relevance of Z in predictive models
  - Conditional expectations represent the best predictions of a variable given certain information
  - in this case Zx is the best prediction of Z given the information in X
- considers interplay between X and Z
- the significance of a variable ( Z) is assessed not only by its direct impact on Y but also by its conditional relationship with X, taking into account the interplay between X and Z
- the reduction in the variables's importance considers both its direct impact and its dependence on X
- ghost variable of Z is any estimator of E(Z|X)
- when Z is not available this E(z|X) allows X to contribute a little bit more in the prediction of Y
- the larger this extra contribution of X, the smaller is the relevance of Z in the prediction of Y
- combines advantages fo LOCO and random permutations
- as replacing Z by E(Z) in f(X,Z) may be more appropriate than replacing it with an independent copy Z' (random permutations)
- replacing Z by E(Z,X) the best prediction of Z given X according quadratic loss
- measures variable relevance by considering the contribution of Z in the prediction of Y when replacing by its ghost variable
- we need to propose a regression model of each explanatory variable over the others and fit all

- relevance of a variable Z, measured by LOCO or by its ghost variable, is proportional to the classical F statistic used for testing H0 : 𝛽Z = 0 against H0 : 𝛽Z ≠ 0
- measuring variable relevance by ghost variables combines the advantages of the other two methods
- when measuring variable relevance by ghost variables we are in some way extending the concept of any variable significance
  - the evaluation extends beyond the univariate relationship and incorporates the conditional context
  - recognizes that a variable's significance is not solely determined by its standalone impact on the target variable but also by its behavior given the values of other relevant variables
  - Ghost variables quantify how well a variable contributes to the prediction of the target variable when considering the context of other variables
- computing condidional expectatations can become computationaly expensive
- ghost variables and knockoffs perform similar to LOCO but are way faster then latter

##### Knockoffs

1. for each variable in X create a corresponding knockoff variable to mimic its statistical properties but independent to response variable
2. fit the model with all original X and X~ knockoff variables
3. by comparing p-Values from original and knockoffs we can control False Discovery Rate (variable should have low p-value in both)

- variables unrelated to the response and that jointly have the same distribution as the original ones, but being as different as possible from them
- they help to controll the false discovery rate in high-dimensional data
- idea: create a set of artificial variable that mimic the statistical properties of the original variables but are known to be unrelated to the response variable
- any realization of the random variable (X~|X=x) can be used as valid knockoff variables
- gaussian knockoffs
- creating knockoffs variables is difficult -> ghost variables are more simple and flexible

  - it can be done by using conditional sampling or model-based methods e.g. gaussian

- lasso estimation to impose a penality on the absolute values of the regression coefficients
  - useful when having a large number of predictors
  - helps in automatic feature selection
  - choice of tuning parameter λ is crucial

#### Importance based on Shapley Values

- provide a fair way of distributing the overall prediction among the individual features
- think of the prediction process as a cooperative game where each feature is a "player"
- consider all feature combinations and compute the model's prediciotn with and without the feature
- average the marginal contributions over all possible permutations
- the Shapley value for each feature is the average contribution it makes across all possible combinations
  - high: indicates that the variable concistently contributes more to model's prediction across different combinations
  - low: variable contribution is less consistent and may be redundant when considering alongside other features
- desirebale properties of Shapley values: fairness, efficiency, and consistency
- R² is equal to the squared sampling correlation coefficient between the observed responses yi and the fitted values ˆy
  - when the p explanatory variables are uncorrelated R²j is the contribution of xj to the global quality measure R² and is a good measure of the relevance of xj in the model
  - but is no longer true when explanatory variables are correlated -> therefore Shapley Values were introduced
- total payoff is v(P)
- question: find a fair distribution of v (P) among the p players to determine the importance of each player in the overall coalition
- desirable properties:
  - efficiency: sum of individual payoffs equals the value of the grand coalition (gain is distributed among players)
  - symmetry: two players i and j are treated equally
  - linearity: when combining 2 coalition games v and w the distributed gains should correspond to the gains derived from v and w
  - null player: payoff of a null payer in a game is zero
- quantity (v (S ∪ {j }) − v (S)) is the marginal contribution of player j to the coalition S
- its Shapley value 𝜙j (v) is the average of these marginal contributions over the possible different permutations of the set P

- for a subset S of p predictors the characterstic function v(S) is the coefficient of R²S in the regression of y against the variables belonging to S
- the Shapley value is a fair distribution of the total R² among the predictors p and measures the importance of the j-th regressor in the model
- this computation of Shapley values is quite time intensive -> average over a moderate number of random permutaiotns of the explanatory variables
- the calculation of the shapley values requires fitting the prediction model as many times as different subset v(Sj(pi)) and v(Sj(𝜋) ∪ {j }) are found
- **large fitting costs**

**marginal contribution**: refers to the additional value or benefit that a particular element contributes when added to a subset of elements.

- features with higher marginal contribution are considered more relevant
- in Shapley value it is averaged over all possible combinations of features to obtain a fair attribution to each feature

#### Partial Dependency Plot PDP

- the PDP corresponding to the j-th variable aims to represent the j-th partial dependence profile function
- therefore it is the graphical representation of the fĵ
- used to understand the marginal effect of a single feature on the model's predictions while keeping other features fixed
- assume independence between the feature of interest and other features

#### Marginal Plots/Local-dependence plots

- plots representing the conditional expectation function
- estimated with a nonparametric regression tool to smooth the scatter plot
- problems of omitted variables can appear

#### Cumulative Local Effects Graph ALE

- computing local effect as partial derivatives which ends in conditional expected local effect of x1 on f
- the ALE plot is the graphical representation of (x1, f1,ALE (x))
- the conditional expected local effect of x1 at f is always equal to 𝛽1, and thus the graph a straight line
- computation is lower then for PDPs

### Local Methods

- providing a explanation for a single prediction f(x) of a non-transparent model f
- common structure: a simple interpretable mehtod g is fitted locally around x that g (x′) ≈ f(x′) when x′ is in a neighborhood of x

#### LIME

- Local Interpretable Model-agnostic Explanations
- d << p easily recognizable properties of x are selected, and their influence in the prediction f (x) is explored
- simple interpretable model g is assumed to take values z which are 0 or 1 (true or false)
- hx is a one-to-one application that is established between elements {0,1}^d and 2^d neighbors
  - e.g when x is a text the selected properties are the presence of d chosen key words in x, hx returns text x without the key words for which zr = 0
- for chosen variables d generate perturbed instance by introducing a small random noise
- obtain predictions for the perturbed instances from the original model
- fit a locally surrogated model to the perturbed instances and their corresponding predictions
- assess performance of the surrogated model to understand how changes in the individual features affect the predictions of selected instance
- the LIME explanation gx for f(x) consists on the selection of K properties among the d that originally were ofu interest, plus the corresponding estimated coefficients
- more a general methodology than a specific method for local explanations

#### Variable Importance by Shapley Value

- a general method for explaining individual predictions of classification models based on Shapley value
- given an instance x the goal is to explain how its feature value (x1,...xp) contribute to the prediction difference between f(x) and the expected prediction if no feature values are known
- 𝜙j (vx ) is effectively measuring how the j-th feature of x is contributing to move the prediciton from the information-less prediction E(f(x)) to the actual prediction f(x)
- for each feature, calculate its marginal contribution to every possible coalition by comparing the prediction with and without the feature.
- the Shapley Value for feature i is given by the average marginal contribution over all permutations
- higher Shapley Values indicate features that consistently contribute more across different combinations of features.

#### SHAP

- = SHapley Additive exPlanations
- unifying six methods including LIME, Shapley value based local explanations
- a number d << p of properties of x are selected -> simple input features
- local one-to-one funcion hx is defined for neighboorhood of x
- an explanation model g(z) is fitted
- the parameters of g(z) are estimated by Shapley values
- it extends the idea of Shapley values to provide additive explanations for the entire model's predictions
- additive property, meaning the contribution of each feature can be interpreted independently of others
-

1. fit model f on training set
2. compute SHAP values for each feature using Shapley values
3. fit a regression model to estimate parameters of g(z) wiht the SHAP values as the dependent variable and corresponding binary vectors as independent variables
4. estimate the regression model -> this estimates represent the contribution of each feature in different feature subsets
5. use the estimated g(z) to evaluate the local variable importance for a specific instance

- Kernel SHAP
  - using LIME framework with quadratic loss, no penalty and Shapley kernel as proximity function
  - by using the Shapley kernel (enforcing local accuracy property)
  - with this LIME reduces to a weighted least square problem that can be solved efficiently

#### Break down Plots

- simplification of Shapley value
- decompose the difference f(x) - E(f(X)) as the sum of p terms, each accounting for the contribution of one of the p coordinates of x
- downside: depends on order of the explanatory variables when interactios between them are present in prediction function
- alternatives:
  - greedy strategy with either step-down or step-up approach
  - average all values across all possible ordering which lead to Shapley values
  - Break-don plots for interactions, are able to capture local interactions between explanatory variables and visualzed them by waterfall plots

1. baseline prediction (model when no features are considered by avering all predictions across all instances in training set)
2. for each feature calculate its contribution to the prediction by adding it to the baseline
3. create a cumulative plot that represents the incremental contribution of adding a specific feature to baseline
4. indicate the direction (negative/positive) whether it decreases or increases the overall prediction
5. sum up all contributions to obtain final prediction for selected instance

#### Indiviudal Conditional Expectation Plots ICE

- a refinement of the PDP
- show the relationship between a specific explanatory variable and the response at the individual level (while PDP does so in an aggregated way)
- shows how the prediction for the i-th case is changing when the value of the j-th predictor Xj is changing gtom the observed value xij to any other possble value z of Xj asumming that others stay constant
- the PDP is the average over all the n data of the individual conditional profiles f^(i)j(z) represented by the ICE plots
- useful is to drawing in gray color the n ICE profiles at the same plot and superimposing in black their average -> the PDP
- the possibility to decompose the global PDP into individual ICE curves is a nice property that is not sared by the ALE plot
