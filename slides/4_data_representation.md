# Data Representation

<br>
<br>
<br>
<br>

<div class="grid grid-cols-[1fr_1fr_2fr_1fr] gap-4">
<div>
<v-click>

#### Feature vector

$\mathbf{x} = \begin{bmatrix}
           x_{1} \\
           x_{2} \\
           \vdots \\
           x_{p}
         \end{bmatrix}$
</v-click>
</div>
<div>
<v-click>

#### Design matrix

$X = \begin{bmatrix}
           \mathbf{x}_{1}^T \\
           \mathbf{x}_{2}^T \\
           \vdots \\
           \mathbf{x}_{N}^T
         \end{bmatrix}$
</v-click>
</div>
<div>
<v-click>

#### $~$
$=~~ {\begin{bmatrix} x_1^1 & x_2^1 & \ldots & x_p^1 \\ x_1^2 & x_2^2 & \ldots & x_p^2 \\ \vdots & \vdots & \ddots & \vdots \\ x_1^N & x_2^N & \ldots & x_p^N \end{bmatrix}}$
</v-click>
</div>
<div>
<v-click>

#### Target vector

$\mathbf{y} = \begin{bmatrix}
           y^1 \\
           y^2 \\
           \vdots \\
           y^N
         \end{bmatrix}$
</v-click>
</div>
</div>

---

# Data Representation

<figure>
  <img src="/iris.png" style="width: 560px !important">
</figure>

---

# Hypothesis Space

<figure>
  <img src="/hypothesis_space.png" style="width: 560px !important">
</figure>

---

# Hypothesis Space Size

<figure>
  <img src="/hypothesis_space_size.png" style="width: 900px !important">
</figure>
<br>

#### How many possible hypotheses?

* $4$ features. For simplicity, assume all of them are binary (True, False)
	* $2^4 = 16$ rules
* $3$ classes labels (Setosa, Versicolor, Virginica)
* Hence, we have $3^{16} = 43~046~721$ potenial combinations
	* This is the size of hypothesis space
