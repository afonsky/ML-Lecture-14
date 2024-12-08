# Classes of Machine Learning Algorithms

* Generalized linear models
  * Linear regression, Logistic regression

* Support vector machines
  * SVC, linear SVM, RBF-kernel SVM

* Tree-based models
  * Decision Trees

* Ensembles
  * Bagging, Boosting, Random forest

* Instance-based learners
  * K-nearest neighbors

---

# Steps for Approaching a ML Application

1. Define the problem to be solved

2. Collect (labeled) data

3. Choose an algorithm class

4. Choose an optimization metric for learning the model

5. Choose a metric for evaluating the model

---

# Objective Functions

<v-clicks depth="3">

* Minimize a mean squared error cost (or loss) function
  * CART, decision tree regression, linear regression, adaptive near neurons, ...
* Maximize information gain/minimize child node impurities
  * CART decision tree classification
* Maximize log-likelihood or minimize cross-entropy loss (or cost) function
* Minimize hinge loss
  * Support vector machines
* Not (yet) covered in our course:
  * Maximize the posterior probabilities (Naive Bayes)
  * Maximize a fitness function (genetic programming)
  * Maximize the total reward/value function (reinforcement learning)
</v-clicks>

---

# Optimization Methods

<v-clicks depth="2">

* Combinatorial search, greedy search
  * Decision trees over, not within nodes
* Unconstrained convex optimization
  * Logistic regression
* Constrained convex optimization
  * SVM
* Nonconvex optimization using backpropagation, chain rule, reverse autodiff
  * Neural networks
* Constrained nonconvex optimization
  * Semi-adversarial networks
</v-clicks>

---

# Evaluation Ex.: Misclassification Error
<br>

#### Loss:<br>
$L(\hat{y},y) = \begin{cases}
0, ~~~\mathrm{if} ~~~\hat{y} = y\\
1, ~~~\mathrm{if} ~~~\hat{y} \neq y
\end{cases}$
<br>
<br>

#### Test error:<br>
$\mathrm{Err}_\mathrm{test} = \Large \frac{1}{n}\sum\limits_{i=1}^n L\big(\hat{y}_i, y_i\big)$

---

# Other Metrics
<br>

* Accuracy (1-Error)
* ROC AUC
* Precision
* Recall
* (Cross) Entropy
* Likelihood
* Squared Error/MSE
* L-norms
* Utility
* Fitness

---

# Categorizing Machine Learning Algorithms

<v-clicks depth="2">

* **Eager vs lazy learners**
  * Eager: process training data immediately
  * Lazy: defer the processing step until the prediction
* **Batch vs online learning**
  * Batch: model is learned on the entire set of training examples
  * Online: earn from one training example at the time
* **Parametric vs nonparametric models**
  * Parametric: we assume a certain functional form for $f(X) = y$
* **Discriminative vs generative**
  * Discriminative: like trying to extract information from text in a foreign language without learning that language
  * Generative: model the joint distribution $P(X,Y) = P(Y)P(X|Y)$ for training pairs
</v-clicks>

---

# Preface to Deep Learning

### In the next Semester we will study:

* Good old Neural Networks, with more layers/modules

* Non-linear, hierarchical, abstract representations of data

* Flexible models with any input/output type and size

* Differentiable Functional Programming

<br>

### You will learn:

* When and where to use DL

* "How" it works

* Frontiers of DL