# Logistic Regression in C

Logistic regression written from scratch in C99, trained on the
[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
dataset to predict whether a customer will subscribe to a term deposit.
Matrix math and memory come from [Sorach](https://github.com/Kenlnw/sorach),
a small linear algebra library included as a git submodule.

## Features

- Batch gradient descent on the cross-entropy loss, with L2 regularization
- **No learning rate to tune**: each epoch picks its own step size from the
  Hessian of the loss (see [The math](#the-math))
- Early stopping once the gradient is close to zero
- 5-fold cross-validation to pick the best regularization strength (lambda)
- Standardization of the numeric columns, with the mean and standard
  deviation saved alongside the weights
- Save a trained model to a binary file and load it back for prediction
- A model summary that lists every learned weight by feature name
- All memory taken from a single arena, freed in one call

## The math

This section follows it and adds the parts the code needs on top
(L2 regularization and how the Hessian is used without building it).

### Notation

Vectors are in bold ($\mathbf{w}$), scalars in italic ($w_j$, $\eta$) and
matrices in uppercase ($X$, $H$). The PDF writes vectors with an arrow
($\vec{w}$) instead.

| Symbol | Meaning | In the code |
|---|---|---|
| $N$ | number of customers (rows) | `X.rows` |
| $d$ | number of features + 1 for the bias | `X.cols` |
| $X \in \mathbb{R}^{N \times d}$ | data matrix, row $i$ is $\mathbf{x}_i^\top$, first column all 1s | `X` |
| $\mathbf{y} \in \{0,1\}^N$ | labels, 1 = subscribed | `y` |
| $\mathbf{w} \in \mathbb{R}^d$ | weights, $w_0$ is the bias | `model->w` |
| $\mathbf{p} \in (0,1)^N$ | predicted probabilities | `p` |
| $\lambda$ | L2 regularization strength | `model->lambda` |
| $\eta$ | learning rate (step size) | `model->lr` |

### 1. Model: sigmoid of a linear function

```math
\mathbf{z} = X\mathbf{w}, \qquad
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad
p_i = P(y_i = 1) = \sigma(z_i), \quad P(y_i = 0) = 1 - \sigma(z_i)
```

A customer is predicted as **YES** when $p_i \ge 0.5$ (`DEFAULT_THRESHOLD`).

### 2. Loss: cross-entropy

```math
L(\mathbf{w}) = -\frac{1}{N}\sum_{i=1}^{N}
\Big[\, y_i \log \sigma(z_i) + (1 - y_i)\log\big(1 - \sigma(z_i)\big) \Big]
```

To keep $\log$ finite, the probabilities are clamped to
$[10^{-7},\ 1 - 10^{-7}]$ before they are used.

### 3. Gradient

The derivative of the sigmoid is

```math
\frac{\partial \sigma(z_i)}{\partial w_j}
= \frac{e^{-z_i}}{(1 + e^{-z_i})^2}\, x_{ij}
= \sigma(z_i)\big(1 - \sigma(z_i)\big)\, x_{ij}
```

Putting it into the chain rule for $L$, the $\sigma(1-\sigma)$ factors cancel
against the $1/\sigma$ and $1/(1-\sigma)$ from the logs, which leaves

```math
\frac{\partial L}{\partial w_j} = -\frac{1}{N}\sum_{i=1}^{N}\big(y_i - \sigma(z_i)\big)\, x_{ij}
\qquad\Longrightarrow\qquad
\nabla L(\mathbf{w}) = \frac{1}{N} X^\top\big(\mathbf{p} - \mathbf{y}\big)
```

### 4. Hessian

Differentiating the gradient once more:

```math
H_{jk} = \frac{\partial^2 L}{\partial w_j\,\partial w_k}
= \frac{1}{N}\sum_{i=1}^{N} \sigma(z_i)\big(1 - \sigma(z_i)\big)\, x_{ij}\, x_{ik}
```

Every $H_{jk}$ is one entry of a sum of outer products, so the whole matrix is

```math
H = \frac{1}{N}\sum_{i=1}^{N} \sigma(z_i)\big(1-\sigma(z_i)\big)\,\mathbf{x}_i\mathbf{x}_i^\top
  = \frac{1}{N} X^\top S X,
\qquad S = \mathrm{diag}\!\big(p_i(1 - p_i)\big)
```

Since every $p_i(1-p_i) > 0$, $H$ is positive semi-definite: the loss is
convex, so gradient descent cannot get stuck in a bad local minimum.

### 5. Choosing the learning rate from the Hessian

Gradient descent updates the weights with

```math
\mathbf{w}^{*} = \mathbf{w} - \eta\, \mathbf{g}, \qquad \mathbf{g} = \nabla L(\mathbf{w})
```

Instead of a fixed $\eta$, approximate the loss around $\mathbf{w}$ with its
second-order Taylor series

```math
L(\mathbf{w}^{*}) \approx L(\mathbf{w}) + (\mathbf{w}^{*} - \mathbf{w})^\top \mathbf{g}
+ \tfrac{1}{2}(\mathbf{w}^{*} - \mathbf{w})^\top H\, (\mathbf{w}^{*} - \mathbf{w})
```

and substitute $\mathbf{w}^{*} - \mathbf{w} = -\eta\,\mathbf{g}$:

```math
L(\mathbf{w} - \eta\mathbf{g}) \approx L(\mathbf{w}) - \eta\,\mathbf{g}^\top\mathbf{g} + \tfrac{1}{2}\eta^2\,\mathbf{g}^\top H \mathbf{g}
```

This is a parabola in $\eta$. The best step is where its derivative is zero:

```math
\frac{d}{d\eta}L(\mathbf{w} - \eta\mathbf{g}) = -\mathbf{g}^\top\mathbf{g} + \eta\,\mathbf{g}^\top H\mathbf{g} = 0
\qquad\Longrightarrow\qquad
\boxed{\ \eta^{*} = \frac{\mathbf{g}^\top\mathbf{g}}{\mathbf{g}^\top H\,\mathbf{g}}\ }
```

So the step is long where the loss is flat (small curvature along
$\mathbf{g}$) and short where it is steep, and it is recomputed every epoch.
If $\mathbf{g}^\top H \mathbf{g} \le 10^{-12}$ (the loss is almost flat in that
direction), the code falls back to $\eta = 1$.

This is still gradient descent, not Newton's method. Newton's method steps
by $H^{-1}\mathbf{g}$, which needs the whole $d \times d$ Hessian and a linear
solve. Here only the scalar $\mathbf{g}^\top H\mathbf{g}$ is needed.

### 6. Computing $H\mathbf{g}$ without building $H$

$H$ is never stored. The product $H\mathbf{g}$ is computed right to left with
matrix-vector products only:

```math
H\mathbf{g} = \frac{1}{N} X^\top\Big( S\,\big(X\mathbf{g}\big) \Big)
```

1. `Xg  = X @ g` (length $N$)
2. `SXg[i] = p[i] * (1 - p[i]) * Xg[i]` ($S$ is diagonal, so this is element-wise)
3. `Hg  = X.T @ SXg / N` (length $d$)

That costs $O(Nd)$ per epoch, the same as the gradient, instead of the
$O(Nd^2)$ it would take to form $X^\top S X$.

### 7. L2 regularization

To keep the weights small, the training loss adds a ridge penalty on every
weight except the bias $w_0$:

```math
L_\lambda(\mathbf{w}) = L(\mathbf{w}) + \frac{\lambda}{2}\sum_{j=1}^{d-1} w_j^2
```

It adds $\lambda w_j$ to each gradient entry and $\lambda$ to the diagonal
of the Hessian (again skipping the bias):

```math
\mathbf{g} = \frac{1}{N}X^\top(\mathbf{p} - \mathbf{y}) + \lambda\tilde{\mathbf{w}}, \qquad
H\mathbf{g} = \frac{1}{N}X^\top S X\mathbf{g} + \lambda\tilde{\mathbf{g}}
```

where $\tilde{\mathbf{w}}$ and $\tilde{\mathbf{g}}$ are $\mathbf{w}$ and $\mathbf{g}$
with the bias entry set to 0. With $\lambda > 0$ the curvature is at least
$\lambda$ in every non-bias direction, which also keeps $\eta^{*}$ bounded.

### 8. One training epoch

Here is one epoch of `model_fit` written in NumPy, since it is shorter and
easier to read than the C version. `@` is matrix multiplication, `X.T` is
the transpose, and `*`, `/`, `-` work element-wise. `w`, `y`, `p` and `g`
are 1-D arrays, so `g @ g` is the dot product $\mathbf{g}^\top\mathbf{g}$.

```python
p = np.clip(sigmoid(X @ w), 1e-7, 1 - 1e-7)

w_reg = w.copy(); w_reg[0] = 0               # no penalty on the bias
g = X.T @ (p - y) / N + lam * w_reg          # gradient
if np.linalg.norm(g) < 1e-5:
    break                                    # converged

g_reg = g.copy(); g_reg[0] = 0
Hg = X.T @ (p * (1 - p) * (X @ g)) / N + lam * g_reg   # H @ g without building H

gHg = g @ Hg
eta = (g @ g) / gHg if gHg > 1e-12 else 1.0  # step size from the Hessian
w = w - eta * g
```

### 9. Standardization

The 10 numeric columns (age, call duration, ...) are on very different
scales, which makes the loss surface long and narrow and slows gradient
descent. Each one is rescaled to mean 0 and standard deviation 1:

```math
x_{ij} \leftarrow \frac{x_{ij} - \mu_j}{s_j}
```

The bias and the one-hot columns are left as they are. $\mu_j$ and $s_j$
are learned from the training data only and saved with the model, so new
customers are scaled exactly the same way at prediction time.

### 10. Cross-validation for lambda

For each $\lambda \in \{0,\ 10^{-4},\ 10^{-3},\ 10^{-2},\ 10^{-1},\ 1\}$
the shuffled data is split into 5 folds. The model is trained on 4 folds
(standardized with those 4 folds' statistics) and scored with the
cross-entropy on the fifth, for each of the 5 choices of held-out fold:

```math
\text{CV}(\lambda) = \frac{1}{5}\sum_{k=1}^{5} L_{\text{val}}^{(k)}(\mathbf{w}_\lambda^{(k)})
```

The $\lambda$ with the lowest $\text{CV}(\lambda)$ is used for the final
training on the whole dataset.

## Project layout

```
.
├── train.c            cross-validate, train on the full dataset, save the model
├── test.c             load the model and predict for 5 example customers
├── src/
│   ├── model.c/.h     LogReg struct, fit, predict, standardize, save/load, summary
│   ├── cv.c/.h        k-fold cross-validation and lambda search
│   ├── read.c/.h      CSV reading
│   └── customers.h    the 5 example customers used by test.c
├── csv/
│   ├── bank-additional.csv         raw dataset (41,188 rows)
│   └── bank-additional-onehot.csv  same data with categories one-hot encoded
├── libs/sorach/       Sorach (git submodule)
└── Makefile
```

## Getting started

Clone with the submodule:

```sh
git clone --recurse-submodules <this-repo-url>
```

If you already cloned without it:

```sh
git submodule update --init
```

You need `gcc` and `make` (on Windows, MinGW's `mingw32-make` works too).

## Usage

```sh
make          # build build/train and build/test
make train    # build and run training; writes bank_model.bin
make test     # build and run prediction with bank_model.bin
make clean    # remove build files, including Sorach's
```

Add `DOUBLE=1` to any command to use `double` instead of `float`. Those
programs are built into `build_double/`. A model saved by a `float` build
cannot be loaded by a `double` build, and the other way round.

### Training

`make train`:

1. Reads `csv/bank-additional-onehot.csv` and adds a bias column
2. Runs 5-fold cross-validation for each lambda in
   `{0, 1e-4, 1e-3, 1e-2, 1e-1, 1}` and keeps the one with the lowest mean
   validation loss
3. Standardizes the 10 numeric columns
4. Trains on the whole dataset with the best lambda for up to 1000 epochs,
   with the step size chosen from the Hessian every epoch, stopping early
   when the gradient norm drops below `1e-5`. Loss and accuracy are printed
   every 100 epochs.
5. Prints the model summary and saves it to `bank_model.bin`

### Prediction

`make test` loads `bank_model.bin`, standardizes five example customers
from `src/customers.h` with the saved mean and standard deviation, and
prints each prediction with its probability. After the model summary,
the output looks like this (your numbers will differ slightly after
retraining, since the folds are shuffled with a time-based seed):

```
Customer 1 : Kate
retired, previous campaign succeeded, called in October
age 67, call 420 s, contacts this campaign 1, last contacted 6 days ago, previous contacts 2
Subscribe : YES (88.57%)

Customer 2 : Jessie
student, long call in March
age 23, call 610 s, contacts this campaign 1, not contacted before, previous contacts 0
Subscribe : YES (91.61%)

Customer 3 : Lewis
blue-collar worker, short call in May, contacted many times
age 38, call 45 s, contacts this campaign 6, not contacted before, previous contacts 0
Subscribe : NO (0.24%)

Customer 4 : Steve
manager, medium-length call in August
age 45, call 280 s, contacts this campaign 2, not contacted before, previous contacts 1
Subscribe : NO (2.37%)

Customer 5 : Robert
technician, average customer in June
age 34, call 200 s, contacts this campaign 3, not contacted before, previous contacts 0
Subscribe : NO (2.18%)
```

To try your own customers, edit `src/customers.h`. Each row is a 1 for the
bias followed by the 62 features in the same column order as the header of
`csv/bank-additional-onehot.csv`.

## Dataset

S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the
Success of Bank Telemarketing.* Decision Support Systems, 2014.
Available from the UCI Machine Learning Repository.
