# Probabilistic Reduced Rank Regression (PRRR) — Implementation Spec

## Overview

Probabilistic Reduced Rank Regression (PRRR) is a probabilistic extension of Reduced Rank Regression (RRR) that makes the latent communication bottleneck between two neural populations explicit. It is to RRR what PPCA is to PCA.

**Use case:** Modeling communication between two populations of neurons (e.g., from different brain regions recorded with Neuropixels probes). The input population $x_t \in \mathbb{R}^m$ communicates with the output population $y_t \in \mathbb{R}^n$ through a low-dimensional latent signal $z_t \in \mathbb{R}^r$.

**Reference:** Wu & Pillow (2025), "Reduced rank regression for neural communication: a tutorial."

---

## 1. Model Specification

### 1.1 Generative Model (Stochastic Bottleneck)

$$z_t \mid x_t \sim \mathcal{N}(U^\top x_t,\; \Psi)$$
$$y_t \mid z_t \sim \mathcal{N}(V z_t,\; \Sigma)$$

**Parameters:**

| Symbol | Shape | Description |
|--------|-------|-------------|
| $U$ | $m \times r$ | Input communication axes (encoder) |
| $V$ | $n \times r$ | Output communication axes (decoder) |
| $\Psi$ | $r \times r$ | Communication noise covariance |
| $\Sigma$ | $n \times n$ | Private output noise covariance |

**Data matrices** (rows are observations/timepoints):

| Symbol | Shape | Description |
|--------|-------|-------------|
| $X$ | $T \times m$ | Input population activity |
| $Y$ | $T \times n$ | Output population activity |
| $Z$ | $T \times r$ | Latent communication signal (unobserved) |

**Marginal distribution** (integrating out $z_t$):

$$y_t \mid x_t \sim \mathcal{N}(VU^\top x_t,\; V\Psi V^\top + \Sigma)$$

The output covariance decomposes into:
- $V\Psi V^\top$: variance explained by noisy communication
- $\Sigma$: private output noise

**Low-rank constraint:** The effective weight matrix $W = VU^\top$ has rank $\leq r$ by construction (bottleneck architecture), so no explicit rank constraint is needed.

### 1.2 Covariance Variants

Support three variants of $\Psi$ (communication noise):

1. **Isotropic:** $\Psi = \psi^2 I_r$ (single scalar)
2. **Diagonal:** $\Psi = \text{diag}(\psi_1^2, \ldots, \psi_r^2)$ (per-dimension, enables ARD)
3. **Full:** $\Psi \in \mathbb{R}^{r \times r}$ (full covariance)

Support two variants of $\Sigma$ (output noise):

1. **Isotropic:** $\Sigma = \sigma^2 I_n$
2. **Diagonal:** $\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_n^2)$
3. **Full:** $\Sigma \in \mathbb{R}^{n \times n}$

Default: diagonal $\Psi$, diagonal $\Sigma$.

### 1.3 Deterministic Bottleneck (Special Case)

When $\Psi \to 0$, the model collapses to standard RRR:

$$z_t = U^\top x_t$$
$$y_t \mid z_t \sim \mathcal{N}(V z_t, \Sigma)$$

This should be verified in tests.

---

## 2. EM Algorithm

### 2.1 E-Step

Compute the posterior $p(z_t \mid x_t, y_t, \theta^{\text{old}})$, which is Gaussian:

**Posterior covariance** (same for all $t$):

$$\Sigma_{z|y,x}^{-1} = \Psi^{-1} + V^\top \Sigma^{-1} V$$

**Posterior mean** (per timepoint):

$$\hat{z}_t = \Sigma_{z|y,x} \left( \Psi^{-1} U^\top x_t + V^\top \Sigma^{-1} y_t \right)$$

**Sufficient statistics needed for M-step:**

$$\hat{Z} \in \mathbb{R}^{T \times r}, \quad \text{rows are } \hat{z}_t^\top$$

$$\hat{P} = \sum_{t=1}^T \mathbb{E}[z_t z_t^\top] = T \cdot \Sigma_{z|y,x} + \hat{Z}^\top \hat{Z}$$

Note: $\hat{P}$ is $r \times r$.

### 2.2 M-Step

Derived from the Q-function (expected complete-data log-likelihood):

$$Q_{U,\Psi} = -\frac{T}{2}\log|\Psi| - \frac{1}{2}\text{Tr}\left(\Psi^{-1}\left[(\hat{Z} - XU)^\top(\hat{Z} - XU) + T\Sigma_{z|y,x}\right]\right)$$

$$Q_{V,\Sigma} = -\frac{T}{2}\log|\Sigma| - \frac{1}{2}\text{Tr}\left(\Sigma^{-1}\left[(Y - \hat{Z}V^\top)^\top(Y - \hat{Z}V^\top) + V \cdot T\Sigma_{z|y,x} \cdot V^\top\right]\right)$$

**Update for $U$:**

$$U^{\text{new}} = (X^\top X)^{-1} X^\top \hat{Z}$$

This is least squares: regress $\hat{Z}$ on $X$.

**Update for $V$:**

$$V^{\text{new}} = Y^\top \hat{Z} \cdot \hat{P}^{-1}$$

Note: uses $\hat{P}$ (which includes the posterior uncertainty correction), not $\hat{Z}^\top \hat{Z}$.

**Update for $\Psi$:**

$$\Psi^{\text{new}} = \frac{1}{T}\left[(\hat{Z} - XU)^\top(\hat{Z} - XU) + T\Sigma_{z|y,x}\right]$$

Two terms: empirical residual covariance + posterior uncertainty correction. If isotropic: take trace and divide by $r$. If diagonal: take diagonal.

**Update for $\Sigma$:**

$$\Sigma^{\text{new}} = \frac{1}{T}\left[(Y - \hat{Z}V^\top)^\top(Y - \hat{Z}V^\top) + V \cdot T\Sigma_{z|y,x} \cdot V^\top\right]$$

Same structure: empirical residual + propagated uncertainty. If isotropic: trace / $n$. If diagonal: take diagonal.

**Key insight:** The $T\Sigma_{z|y,x}$ terms prevent underestimation of noise. Without them, we'd pretend $\hat{z}_t$ is the true $z_t$ and systematically undercount variance.

### 2.3 Log-Likelihood for Monitoring

Use the marginal log-likelihood (with $z$ integrated out) to monitor convergence:

$$\log p(Y \mid X, \theta) = -\frac{T}{2}\log|V\Psi V^\top + \Sigma| - \frac{1}{2}\sum_{t=1}^T (y_t - VU^\top x_t)^\top (V\Psi V^\top + \Sigma)^{-1} (y_t - VU^\top x_t) + \text{const}$$

This should monotonically increase across EM iterations.

---

## 3. Prediction and Uncertainty Quantification

### 3.1 Point Prediction

$$\hat{y}_t = V U^\top x_t$$

(Same as standard RRR.)

### 3.2 Predictive Variance

For a new input $x_*$, the predictive distribution is:

$$y_* \mid x_* \sim \mathcal{N}(VU^\top x_*,\; V\Psi V^\top + \Sigma)$$

So per-output predictive std is $\text{diag}(V\Psi V^\top + \Sigma)^{1/2}$.

### 3.3 Scoring

Use $R^2$ for comparison with standard RRR:

$$R^2 = 1 - \frac{\sum_t \|y_t - \hat{y}_t\|^2}{\sum_t \|y_t - \bar{y}\|^2}$$

Also compute marginal log-likelihood for model comparison across ranks.

---


