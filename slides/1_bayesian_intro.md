---
layout: center
---
# Bayesian inference


---
zoom: 0.90
---

# Algorithms & Inference

<v-clicks depth="2">

* **Algorithm** -- set of instructions to compute some statistics
* **Inference** -- reasoning about the importance of these statistics

* Eg. Consider final exam grades (in $\%$ units) of $n = 100$ students: $x_{1:100}$

* **Mean** summarizes the observed sample as single number $\bar{x} := \frac{1}{n} \sum x_i$
* **Standard Error (SE)** measures the accuracy of $\bar{x}$: $~~~~~~~~~\widehat{\mathrm{se}}^2 := \frac{1}{n} \cdot \color{grey}\underbrace{\color{#006}\frac{1}{n-1} \sum_i (x_i - \bar{x})^2}_{\mathrm{unbiased~variance~of~} x_i}$
  * Sample variance $\rightarrow$ true variance, assuming are i.i.d. $x_i$
  * $\widehat{\mathrm{se}}(\bar{x}) \xrightarrow[]{n\to\infty} 0$
* An estimate and its accuracy is reported as $\bar{x} \pm \widehat{\mathrm{se}}(\bar{x}) \cdot 1$
  * This is an example of **frequentist's inference**
* Calculations of $\bar{x}, \widehat{\mathrm{se}}(\bar{x})$ are algorithmic but their interpretations are inferential
</v-clicks>

---

# Foundation of Frequentism

<div style="font-size:18px; ">
<v-clicks depth="3">

* In statistical inference the observations $x_{1:n}$ are assumed to be drawn from distribution $\mathcal{F}$
  * $x_{1:n}$ are realizations of random variables (RV) $X := X_{1:n}$ where $X_i \stackrel{iid}{\sim} \mathcal{F}$
    * Since all $X_i$ are assumed i.i.d., we replace them with just $X_i \sim \mathcal{F}$
  * Recall: $x_i := X(\omega_i)$, where $\omega_i \in \Omega$  is an outcome from a sample space
  * So, $X: \Omega \to \R$, i.e. an RV is a real-valued function on sample space
* Often we would like to predict the value of a single draw
  * That is we want to know $\theta := \mathbb{E}_\mathcal{F} X = \mathbb{E}_\mathcal{F} \bar{X}$, $\forall n$ with natural (algorithmic) estimate $\hat\theta := \bar{x}$
  * $\hat\theta$ has many "nice" properties, such as $\hat\theta \xrightarrow[]{n\to\infty} \theta$
* In frequentism:
  * $\bar{x}$ is a single real number
    * In a multivariate setting, $\bar{x}$ is a centroid or mean vector
  * $\bar{X}$ is an RV with its own distribution
  * We summarize (or decompose) the accuracy of $\mathbb{E}\bar{X}$ with a **bias** and **variance**
</v-clicks>
</div>

---

# Frequentism

<div style="font-size:19px; ">
<v-clicks depth="3">

* Probabilistic properties of $X$ are derived theoretically and then computed from a sample
* **Problem**: the true distribution family, $\mathcal{F}$, is unknown
  * So, we use algorithms to avoid dependence on $\mathcal{F}$
  * **Plug-in Principle**: express an estimator in terms of a large sample and plug-in observations
    * $\widehat{\mathbb{E}X} = \frac{1}{n} \sum x_i$ or $\widehat{\mathbb{V}X} = \frac{1}{n-1} \sum x_i$
  * **Taylor-Series Approximation** (**Delta Method**): we estimate more complex statistics using first order derivative or linear approximation at a point
  * **Maximum Likelihood Estimation** (**MLE**): we assume a parametric distribution $\mathcal{F}$ and estimate its parameters
    * i.e. we pick the best estimation algorithm for the given $\mathcal{F}$
  * **Simulation and Bootstrap**: draw $B$ samples (with replacement) from the sample $\mathcal{x}$, compute statistics of interest $B$ times and compute its bootstrapped variance
  * **Pivotal statistics** do not depend on the underlying distribution $\mathcal{F}$
</v-clicks>
</div>

---

# Bayesian Inference

<div style="font-size:19px; ">
<v-clicks depth="3">

* In Bayesian Inference we assume
  * Probability density family parameterized with $\theta, \mathcal{F}_{\theta}(X)$
    * $\theta$ can be a scalar or a vector. Ex.: $\mathrm{Multinomial}(\theta_1, ..., \theta_{p-1})$
  * A prior density, $f(\theta)$: prior information about the parameter $\theta$
    * $f$ is available to us **before** we observe the data $x$ (note no dependence on $X$)

* Ex.
  * $\mathcal{F}_{\theta}(X) = \bigg\{\frac{1}{\sqrt{2\pi}} e^{-0.5(x-\theta)^2}\bigg\}_\theta$, Gaussian density with mean $\theta$ and variance $1$
  * If we assume (from some knowledge) $\theta$ is always positive, but never exceeds $2$, then we might assume $f(\theta) = \mathrm{Unif}(\theta|0, 2) = \frac{1}{2}$ or some other distribution with support $[0, 2]$
</v-clicks>
</div>

---

# Bayes Theorem
<div style="font-size:19px; ">
<v-clicks depth="3">
<div class="grid grid-cols-[2fr_1fr] gap-5">
<div>

* 250 years old formula

$$
\color{grey}\underbrace{\color{#006} f(\theta | x)}_{\mathrm{posterior}} \color{#006} =
\color{grey}\underbrace{\color{#006} f(\theta)}_{\mathrm{prior}} \color{#006} \cdot
\color{grey}\underbrace{\color{#006} f(x | \theta)}_{\mathrm{likelihood}} \color{#006} \bigg/
\color{grey}\underbrace{\color{#006} f(x)}_{\mathrm{marginal}} \color{red} \propto
\color{grey}\underbrace{\color{#006} f(\theta) \cdot f_\theta(x)}_{\mathrm{unnormalized~ posterior}}
\color{#006},
$$
where the marginal density is a constant w.r.t. $\theta$
</div>
<div>
<figure>
  <img src="/Thomas_Bayes.gif" style="width: 135px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px">Thomas Bayes (1702—1761)
  </figcaption>
</figure>
</div>
</div>

* **Goal**: infer the posterior distribution of $\theta | x$ from the assumed prior $f$ and the likelihood $f_\theta$
  * It combines **prior** knowledge about $\theta$ with cthe current  **evidence** (data) to improve the posterior
  * Here $x$ is fixed and $\theta$ and $\theta | x$ are RVs
* If we know the density $f(\theta) \cdot f_\theta(x)$, which must integrate to $1$, then:
  * The constant of integration, $f(x)$, can be determined numerically
  * Likewise, we can drop all constants (w.r.t. $\theta$) in the likelihood and prior to simplify the expression
</v-clicks>
</div>

---

# Effect of Observed Sample
<div></div>

$$
\color{grey}\underbrace{\color{#006} f(\theta | x)}_{\mathrm{posterior}} \color{red} \propto
\color{grey}\underbrace{\color{#006} f(\theta) \cdot f_\theta(x)}_{\mathrm{prior} \cdot \mathrm{likelihood}}
$$

<div class="grid grid-cols-[1fr_1fr] gap-5">
<div>
<v-clicks depth="3">

* A prior is our guess about the posterior
  * We can make pedictions with no observation
    * In frequentist world, this is not possible
* With observed sample we update the posterior and improve predictions
* Posterior is a pointwise product
</v-clicks>
</div>
<div>
<figure>
  <img src="/prior2post_1-1.svg" style="width: 450px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px">Image source:
    <a href="https://m-clark.github.io/bayesian-basics/example.html">https://m-clark.github.io/bayesian-basics/example.html</a>
  </figcaption>
</figure>
</div>
</div>

---
zoom: 0.96
---

# Ex. Spelling Correction ([see full example](http://www.stat.columbia.edu/~gelman/stuff_for_blog/spelling.pdf))

<v-clicks depth="3">

* Suppose a text has a word "$\mathrm{radom}$". Is this correct or should it be "$\mathrm{random}$" or "$\mathrm{radon}$"?
* We can compute the posterior probability of $\theta \in \{\mathrm{random, radon, radom}\}$ as
$$
\mathbb{P}[\theta | x = \mathrm{radom}] \color{red} \propto \color{#006}
\mathbb{P}\theta \cdot \mathbb{P}[x = \mathrm{radom}]
$$
* Prior $\mathbb{P}\theta$ can be estimated from fraction of use of each of these words in some text corpus
  * $\mathbb{P}[\theta = \mathrm{random}] = 8 \cdot 10^{-5}$
  * $\mathbb{P}[\theta = \mathrm{radon}] = 6 \cdot 10^{-6}$
  * $\mathbb{P}[\theta = \mathrm{radom}] = 3 \cdot 10^{-7}$, which can be our proior for any non-existing word
* Likelihood of typing "$\mathrm{radom}$" given that some value $\theta$ was intended is:<br>
$\mathbb{P}[x = \mathrm{radom}] = \begin{cases}
        0.00193, \mathrm{\textcolor{grey}{~a~typo}}\\
        0.000143, \mathrm{\textcolor{grey}{~a~typo}}\\
        0.975, \mathrm{\textcolor{grey}{~correct}}\\
      \end{cases}$
* We can normalize the posterior at each $\theta$ or just pick a corrected word as $\argmax\limits_{\theta} \mathbb{P}[\theta | x = \mathrm{radom}]$
</v-clicks>

---

# Informative Priors

<v-clicks depth="4">

* We use historical knowledge about the distribution of the parameter $\theta$

<div class="grid grid-cols-[4fr_2fr] gap-8">
<div>

* **Strong prior**:
  * We consider the distribution of $\theta$ fully specified
  * Ex. $\theta \sim \mathcal{N}(0, 1)$
</div>
<div>
<br>
<br>

  * $\vec{\theta} \sim
  \begin{bmatrix}
  \mathrm{Beta}(1, 2)\\
  \chi_3^2
  \end{bmatrix}$
</div>
</div>
<br>

* **Weak (or hierarchical) prior**:
  * Distribution of $\theta$ is only **partially** specified and remaining information is captured hierarchically
  * Eg. $\theta \sim \mathcal{N}(\mu, 1)$ with $\mu \sim \mathrm{Gamma}(1, 3)$
    * This hierarchy can go deeper, if needed:
      * $\theta \sim \mathcal{N}(\mu, 1)$ with $\mu \sim \mathrm{Gamma}(k, 3)$, where $k \sim \mathrm{Beta}(\alpha, \beta)$, ...
</v-clicks>

---
zoom: 0.97
---

# Uninformative (or Flat or Noninformative) Prior

<v-clicks depth="4">

* Useful when we have **no opinion** about prior distribution of $\theta$
  * Flat prior has smaller effect on the posterior
* We impose no distribution on $\theta$ by assuming $\theta \sim \mathrm{Unif}(a, b) = \frac{1}{b - a}, ~a < b$
<div class="grid grid-cols-[4fr_3fr] gap-5">
<div>

* **Proper prior**:
  * $a, b$ are finite
    * Eg. $\theta \sim \mathrm{Unif}(0, 10)$, a uniform prior
  * i.e. $f(\theta)$ integrates to $1$
* **Improper prior**:
  * $\theta$ has infinite support
    * Ex. $\theta \sim \mathrm{Unif}(-\infty, \infty)$
  * Then $\int_\mathbb{R} f(\theta)d\theta = \infty$
    * Posterior **may** still be a valid PDF or PMF
  * **Jeffreys' prior**: coming up...
</div>
<div>
  <br>
<figure>
  <img src="/uniform_prior.png" style="width: 380px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px">Image source:
    <a href="https://www.quora.com/What-is-an-example-of-a-uniform-prior">https://www.quora.com/What-is-an-example-of-a-uniform-prior</a>
  </figcaption>
</figure>
</div>
</div>
</v-clicks>

---

# Ex. Binomial PMF with a Flat Prior

* Let $x$ be the number of girls in $n$ recorded births
  * i.e. the likelihood is a $\mathrm{Binomial}(x| n, \theta) = C_x^n \theta^x (1 - \theta)^{n-x} \cdot \mathbb{1}_{x \in [0,...,n]} \cdot \mathbb{1}_{\theta \in [0,1]}$
    * Recall: sum of $n$ i.i.d. $\mathrm{Bernoulli}(\theta)$ is a $\mathrm{Binomial}(n, \theta)$
* It is accepted that females are born $48.5\%$ of the time
  * i.e. the prior is flat, $f(\theta) = 0.485$, which is a constant w.r.t. $\theta$
<div class="grid grid-cols-[2fr_1fr] gap-5">
<div>

* Then the posterior for $\theta$:<br>
$f(\theta|x) \color{red} \propto \color{#006} \theta^x (1 - \theta)^{n-x} \color{red} \propto \color{#006} \mathrm{Beta}(\theta | x + 1, n - x + 1)$
* Larger sample yields a more precise posterior
  * Even though the fraction of girl births is the same
</div>
<div>
<figure>
  <img src="/binomial_PMF.png" style="width: 270px !important">
</figure>
</div>
</div>

---

# Ex. Binomial PMF with a Beta Prior

* Consider likelihoods and priors that are mathematically convenient in derivation of a posterior. Consider a Beta prior:
  * $\mathrm{Binomial}(x| n, \theta) \propto \theta^x (1 - \theta)^{n-x}$
  * $\mathrm{Beta}(\alpha, \beta) \propto \theta^{\alpha - } (1 - \theta)^{\beta-1}$

Then:<br>
$~~~~f(\theta | x) \propto \theta^x (1 - \theta)^{n-x} \cdot \theta^{\alpha - 1} (1 - \theta)^{\beta-1}\\
~~= \theta^{x + \alpha - 1} (1 - \theta)^{n - x + \beta - 1}\\
~~= \mathrm{Beta}(\theta ~|~ \alpha + x, ~\beta + n - x)$

* If the posterior and prior are from the same distribution family, we call them **conjugates** (for the given likelihood family)
  * Conjugate families are mathematically tractable

---

# Statistics of the Posterior

* Finally, we can summarize the posterior in terms of **mean** and **variance**

* Continued example for Beta conjugates given the Binomial likelihood, we compute:

  * $\mathbb{E}[\theta | x] = \frac{\alpha + x}{\alpha + \beta + n}$ is a mean for the $\mathrm{Beta}(\theta ~|~ \alpha + x, ~\beta + n - x)$

  * $\mathbb{V}[\theta | x] = \frac{(\alpha + x)(\beta + n - x)}{(\alpha + \beta + n)^2 (\alpha + \beta + n + 1)}$

---
zoom: 0.97
---

# Jeffreys' Prior (see [CASI, p. 29](https://hastie.su.domains/CASI_files/PDF/casi.pdf#page=47))

* Jeffreys' prior of $\theta$ is **invariant** to $1-1$ parameterization of $\theta, \phi := h(\theta)$, for some $h()$
  * i.e. $f(\theta)$ is equivalent (in some sense) to $f(\phi)$
* We can compute the prior of $\phi$ as
$
f(\phi) = f(\theta) \big| \frac{d\theta}{d\phi} \big| = f(\theta) |h^{\prime}(\theta)|^{-1}
$

* Also, via Bayes' rule: $f(\theta) = f(x, \phi) / f_\phi(x)$
* Then Jeffreys' prior of $\theta$ leads to a definition: $f(\theta) \propto [J(\theta)]^{\frac{1}{2}}$
  * i.e. $f(\theta)$ is proportional to square root of **Fischer information** of $\theta$, which is
$$
J(\theta) := \mathbb{E} \bigg[ \bigg(\frac{d}{d\theta} \ln f_\theta(x) | \theta \bigg)^2 \bigg]
= -\mathbb{E} \bigg[ \frac{d^2}{d\theta^2} \ln f_\theta(x) | \theta \bigg]
$$
* Jeffreys' prior depends on $f_\theta(x)$
  * See also [prior choice recommendations](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations)

---

# Ex. Jeffreys' Prior for Binomial Distribution

* Let $X \stackrel{iid}{\sim} f_\theta(x) := \mathrm{Bin}(x|n,\theta) := \begin{pmatrix} n \\ x \end{pmatrix} (1 - \theta)^{n - x} \cdot \mathbb{1}_{x \in [0,...,n]} \cdot \mathbb{1}_{\theta \in [0,1]}$ 

* The log-likelihood of $X$ is: <br>
$\ln f_\theta(x) \stackrel{\perp X_i}{=}
\ln \prod_i f_\theta(x_i) \stackrel{i.d. x_i}{=}
\sum_i \ln \mathrm{Bin}(x_i | n, \theta) = \\\mathrm{const.} + (\sum_i x_i) \cdot \ln\theta + (n^2 - \sum_i x_i) \cdot \ln(1 - \theta)$

* Then the **Fischer Info** is then (use $\mathbb{E}X = \mathbb{E}X_i = np$): $J(\theta) \propto \theta^{-1} (1 - \theta)^{-1}$
<div class="grid grid-cols-[3fr_2fr] gap-5">
<div>

* Then Jeffreys' prior of $\theta$ is $f(\theta) \propto \theta^{\frac{1}{2}} (1 - \theta)^{\frac{1}{2}}$
  * which is the PDF $\mathrm{Beta}\big( \frac{1}{2}, \frac{1}{2} \big)$
    * where $\mathrm{Beta}(\alpha_{>0}, \beta_{>0}) := \frac{x^{\alpha-1}(1-x)^{\beta-1}}{\mathrm{B}(\alpha, \beta)}$
      * where Beta function is $\mathrm{B}(\alpha, \beta) := \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$
        * where $\Gamma(z) := \int_0^{\infty} \nu^{z-1} \mathrm{e}^{-\nu}d\nu$
</div>
<div>
<figure>
  <img src="/binomial_Jeffreys.png" style="width: 450px !important">
</figure>
</div>
</div>

---

# Literature and Links

* This week's topic is covered in [CASI](https://hastie.su.domains/CASI_files/PDF/casi.pdf) (Chapter 5) and [BDA](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf) textbooks

<br>

<div class="grid grid-cols-[1fr_1fr] gap-5">
<div>
<figure>
  <img src="/CASI_cover.jpg" style="width: 200px !important;">
</figure>
</div>
<div>
<figure>
  <img src="/BDA_cover.jpeg" style="width: 210px !important;">
</figure>
</div>
</div>

<br>

* See also [Bayesian Data Analysis course](https://avehtari.github.io/BDA_course_Aalto) by Aki Vehtari
