---
layout: section
---

# Overview of Statistical Learning


---
layout: quote
---

## *Programming computers to learn from experience should eventually eliminate the need for much of this detailed programming effort.*
<br>
<br>
<div style="text-align: right"> Arthur L. Samuel (1959)*<br> AI & ML pioneer</div>

<br>
<br>
<br>
<br>
<br>
<br>
<br>

##### * Arthur L. Samuel, “Some studies in machine learning using the game of checkers", *IBM Journal of research and development* 3.3 (1959), pp. 210-229

---

## The Traditional Programming Paradigm
<br>
<v-click>

```mermaid {scale: 1.2}
stateDiagram
    direction LR
    Observations --> Programmer
    Observations --> Program
    Programmer --> Program
    Program --> Computer
    Computer --> Outputs
    Outputs --> Programmer
```
</v-click>
<br>
<br>
<v-click>

## Machine Learning Paradigm
<br>

```mermaid {scale: 1.2}
stateDiagram
    direction LR
    Observations --> Computer
    Outputs --> Computer
    Computer --> Program
```
##### *Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed*
</v-click>

---
layout: quote
---

## *A computer program is said to **learn** from experience $E$<br> with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$,<br> improves with experience $E$.*
<br>
<br>
<div style="text-align: right"> Tom Mitchell (1997)*<br> Machine Learning Professor<br> at Carnegie Mellon University </div>

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

##### * Tom M Mitchell et al., “Machine learning”, *Burr Ridge, IL: McGraw Hill 45.37* (1997), pp. 870–877

---
layout: quote
---

## *A computer program is said to **learn** from experience $E$<br> with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$,<br> improves with experience $E$.*
<br>

### Consider the problem of recognizing handwritten digits

<figure>
  <img src="/digits.png" style="width: 900px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px;">Examples of digits from the MNIST dataset
  </figcaption>
</figure>

Here:
* Task $T$: classifying handwritten digits from images
* Performance measure $P$: percentage of digits classified correctly
* Training experience $E$: dataset of digits given classifications, e.g., MNIST

---

# Applications of Machine Learning

<div class="grid grid-cols-[1fr_1fr] gap-5">
<div>

* Email spam detection
* Face detection and matching
  * Unlocking your phone
* Web search
* Post office
  * Sorting letters by zip codes
* Credit card fraud
* Stock predictions
* Smart assistants
  * Apple Siri, Yandex Alisa
</div>
<div>

* Product recommendations
  * Netflix, Amazon
* Self-driving cars
  * Uber, Tesla
* Language translation
  * Google translate, DeepL
* Sentiment analysis
* Medical diagnoses
* ...
</div>
</div>

---

# Applications of Machine Learning

* What is the desired outcome?
* What could the dataset look like?
* Is this a supervised or unsupervised problem, and what algorithms would you use?
* How would you measure success?
* What are potential challenges or pitfalls?

---

# Categories of Machine Learning

<figure>
  <img src="/ml_categories.png" style="width: 600px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:<br> <a href="https://github.com/rasbt/python-machine-learning-book-2nd-edition">S.Raschka, V. Mirjalili: Python Machine Learning, 2nd Ed.</a>
  </figcaption>
</figure>

---

# Supervised Learning: Classification

<figure>
  <img src="/ex_classification.png" style="width: 400px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:<br> <a href="https://github.com/rasbt/python-machine-learning-book-2nd-edition">S.Raschka, V. Mirjalili: Python Machine Learning, 2nd Ed.</a>
  </figcaption>
</figure>

---

# Supervised Learning: Regression

<figure>
  <img src="/ex_regression.png" style="width: 400px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:<br> <a href="https://github.com/rasbt/python-machine-learning-book-2nd-edition">S.Raschka, V. Mirjalili: Python Machine Learning, 2nd Ed.</a>
  </figcaption>
</figure>

---

# Unsupervised Learning: Clustering

<figure>
  <img src="/ex_unsupervised.png" style="width: 400px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:<br> <a href="https://github.com/rasbt/python-machine-learning-book-2nd-edition">S.Raschka, V. Mirjalili: Python Machine Learning, 2nd Ed.</a>
  </figcaption>
</figure>

---

# Unsupervised Learning: Dimensionality Reduction

<br>
<br>
<figure>
  <img src="/ex_unsupervised_dim_reduction.png" style="width: 900px !important">
</figure>

---

# Reinforcement Learning

<br>
<figure>
  <img src="/rl.png" style="width: 400px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:<br> <a href="https://github.com/rasbt/python-machine-learning-book-2nd-edition">S.Raschka, V. Mirjalili: Python Machine Learning, 2nd Ed.</a>
  </figcaption>
</figure>
<br>
<v-click>

Applications:
* Self-driving cars
* AlphaGo / AlphaZero
* Auto-GPT
</v-click>

---

# Supervised Learning Problem

* Let $X$ be a **data matrix** with $N$ observations and $p$ predictors:<br>
$X = [X_1, X_2, ..., X_p] \in \R^{N \times p}$

* Let $Y$ be the **response** vector with $N$ corresponding labels:<br>
$Y \in R^N$

* We want to find a **fixed**, but **unknown** function $f$:<br>
$Y = f(X) + \varepsilon$
  * Here we have $\varepsilon$, which is an **independent**, **zero-mean** and **homoscedastic** random variable: $\varepsilon \perp\!\!\!\!\perp X_i, \mathbb{E}\varepsilon = 0, \mathbb{V}\varepsilon < \infty$

* The goal in statistical learning is then to estimate $f$, which can then be used to predict $Y$:<br>
$\hat{f}(X) = \hat{Y}$
  * $\hat{f}$ is also known as the **hypothesis** function

---

# Supervised Learning Workflow

<br>
<figure>
  <img src="/ml_workflow_1.png" style="width: 500px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:<br> <a href="https://github.com/rasbt/python-machine-learning-book-2nd-edition">S.Raschka, V. Mirjalili: Python Machine Learning, 2nd Ed.</a>
  </figcaption>
</figure>

---

# Supervised Learning Workflow

<figure>
  <img src="/ml_workflow_2.png" style="width: 570px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source: <a href="https://github.com/rasbt/python-machine-learning-book-2nd-edition">S.Raschka, V. Mirjalili: Python Machine Learning, 2nd Ed.</a>
  </figcaption>
</figure>