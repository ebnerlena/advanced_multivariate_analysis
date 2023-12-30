# AMA Notes

## Introduction

Supervised Learning (the prediction problem):

- Regression: Predicting of a quantitative response.
- Classification (or discriminant analysis): Predicting a qualitative variable

Unsupervised Learning (to learn relationships and structure from data):

- Density estimation (histogram, kernel density estimation, ...)
- Clustering (hierarchical clustering, k-means, ...)
- Dimensionality reduction (PCA, MDS, principal curves, ISOMAP, manifold learning, ...),
- Detecting communities in social networks

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

### Histogram

### Kernel Density Estimator

---

## Clustering

### Dbscan

### K-means

---

## Dimensionality Reduction

---

Local regression techniques are particularly useful when dealing with complex, non-linear relationships that may vary across different parts of the dataset.

## Non Parametric Regression

- the goal is to model the relationship between the predictors and the response variable without making strong assumptions about the functional form of that relationship
- allow the data to dictate the shape of the relationship

### Spline Smoothing

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

**extrapolation** involves making predictions for input values that extend beyond the range of the training data

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
