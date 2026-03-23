# Probabilistic Reduced Rank Regression (PRRR) — Implementation Spec

## Overview

Probabilistic Reduced Rank Regression (PRRR) is a probabilistic, dynamical extension of Reduced Rank Regression (RRR) that models communication between two neural populations through a low-dimensional latent signal with temporal dynamics. It combines the communication bottleneck of RRR with the temporal structure of a Linear Dynamical System (LDS).

**Use case:** Modeling communication between two populations of neurons (e.g., from different brain regions recorded with Neuropixels probes). The input population $x_t \in \mathbb{R}^m$ communicates with the output population $y_t \in \mathbb{R}^n$ through a low-dimensional latent signal $z_t \in \mathbb{R}^r$ that evolves over time with its own dynamics.

**Reference:** Wu & Pillow (2025), "Reduced rank regression for neural communication: a tutorial."

---

## 1. Model Specification

### 1.1 Generative Model (Input-Driven LDS with Communication Bottleneck)

**Initial condition:**

$$z_1 \sim \mathcal{N}(\mu_0,\; \Sigma_0)$$

**Latent dynamics** (for $t \geq 2$):

$$z_t \mid z_{t-1}, x_t \sim \mathcal{N}(A z_{t-1} + U^\top x_t,\; \Psi)$$

**Observation model:**

$$y_t \mid z_t \sim \mathcal{N}(V z_t,\; \Sigma)$$

The latent $z_t$ integrates two sources of information: its own history (via $A$) and new input from the source population (via $U^\top x_t$). This captures the idea that inter-region communication is not instantaneous and memoryless — the receiving circuit maintains and updates an internal state.

**Parameters:**

| Symbol | Shape | Description |
|--------|-------|-------------|
| $A$ | $r \times r$ | Latent dynamics matrix |
| $U$ | $m \times r$ | Input communication axes (encoder) |
| $V$ | $n \times r$ | Output communication axes (decoder) |
| $\Psi$ | $r \times r$ | Process noise / communication noise covariance |
| $\Sigma$ | $n \times n$ | Observation noise / private output noise covariance |
| $\mu_0$ | $r \times 1$ | Initial latent mean |
| $\Sigma_0$ | $r \times r$ | Initial latent covariance |

**Data matrices** (rows are observations/timepoints):

| Symbol | Shape | Description |
|--------|-------|-------------|
| $X$ | $T \times m$ | Input population activity |
| $Y$ | $T \times n$ | Output population activity |
| $Z$ | $T \times r$ | Latent communication signal (unobserved) |

**Low-rank constraint:** The instantaneous input-to-output mapping $W = VU^\top$ has rank $\leq r$ by construction, same as static PRRR. But now the output also depends on the latent history, so the full temporal receptive field is richer.

### 1.2 Relationship to Standard Models

| Model | Dynamics | Input | Observation |
|-------|----------|-------|-------------|
| Standard LDS | $z_t = A z_{t-1} + w_t$ | None | $y_t = C z_t + v_t$ |
| Input-driven LDS | $z_t = A z_{t-1} + B u_t + w_t$ | Generic input $u_t$ | $y_t = C z_t + v_t$ |
| Static PRRR | None ($z_t = U^\top x_t + w_t$) | Source population $x_t$ | $y_t = V z_t + v_t$ |
| **Dynamic PRRR** | $z_t = A z_{t-1} + U^\top x_t + w_t$ | Source population $x_t$ | $y_t = V z_t + v_t$ |

Dynamic PRRR is an input-driven LDS where $B = U^\top$ and $C = V$, with the constraint that input comes specifically from another neural population through a low-rank encoder. When $A = 0$, it reduces to static PRRR. When $U = 0$, it reduces to a standard LDS on the output population alone.

### 1.3 Covariance Variants

Support three variants of $\Psi$ (process/communication noise):

1. **Isotropic:** $\Psi = \psi^2 I_r$ (single scalar)
2. **Diagonal:** $\Psi = \text{diag}(\psi_1^2, \ldots, \psi_r^2)$
3. **Full:** $\Psi \in \mathbb{R}^{r \times r}$ (full covariance)

Support three variants of $\Sigma$ (observation/output noise):

1. **Isotropic:** $\Sigma = \sigma^2 I_n$
2. **Diagonal:** $\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_n^2)$
3. **Full:** $\Sigma \in \mathbb{R}^{n \times n}$

Default: diagonal $\Psi$, diagonal $\Sigma$.

### 1.4 Special Cases

**Static PRRR** ($A = 0$): Latent has no memory, each timepoint is independent given $x_t$. Reduces to the static model:

$$z_t \mid x_t \sim \mathcal{N}(U^\top x_t, \Psi), \quad y_t \mid z_t \sim \mathcal{N}(V z_t, \Sigma)$$

**Pure LDS** ($U = 0$): No input drive, latent evolves autonomously. Reduces to a standard LDS on the output population.

**Deterministic bottleneck** ($\Psi \to 0$, $A = 0$): Collapses to standard RRR.

---

## 2. EM Algorithm

The E-step is now a Kalman smoother (rather than independent per-timepoint posteriors), and the M-step uses smoother sufficient statistics.

### 2.1 E-Step: Kalman Filter and Smoother

Because the model is a linear-Gaussian state space model, the posterior $p(z_{1:T} \mid x_{1:T}, y_{1:T}, \theta^{\text{old}})$ is Gaussian and can be computed exactly via the Kalman filter (forward pass) followed by the RTS smoother (backward pass).

#### 2.1.1 Forward Pass (Kalman Filter)

Initialize:

$$\hat{z}_{1|0} = \mu_0, \quad P_{1|0} = \Sigma_0$$

For $t = 1, \ldots, T$:

**Observation update (incorporate $y_t$):**

$$K_t = P_{t|t-1} V^\top (V P_{t|t-1} V^\top + \Sigma)^{-1}$$

$$\hat{z}_{t|t} = \hat{z}_{t|t-1} + K_t (y_t - V \hat{z}_{t|t-1})$$

$$P_{t|t} = (I - K_t V) P_{t|t-1}$$

**Prediction (propagate to $t+1$):** For $t < T$:

$$\hat{z}_{t+1|t} = A \hat{z}_{t|t} + U^\top x_{t+1}$$

$$P_{t+1|t} = A P_{t|t} A^\top + \Psi$$

Note: The input $x_{t+1}$ enters at the prediction step. The convention here is that $x_t$ drives the transition *into* time $t$, matching the generative model $z_t \mid z_{t-1}, x_t$.

#### 2.1.2 Backward Pass (RTS Smoother)

Initialize with filter output at $T$:

$$\hat{z}_{T|T}, \quad P_{T|T}$$

For $t = T-1, \ldots, 1$:

$$G_t = P_{t|t} A^\top P_{t+1|t}^{-1}$$

$$\hat{z}_{t|T} = \hat{z}_{t|t} + G_t (\hat{z}_{t+1|T} - \hat{z}_{t+1|t})$$

$$P_{t|T} = P_{t|t} + G_t (P_{t+1|T} - P_{t+1|t}) G_t^\top$$

#### 2.1.3 Sufficient Statistics for M-Step

Define the smoothed quantities:

$$\hat{z}_t \equiv \hat{z}_{t|T} = \mathbb{E}[z_t \mid x_{1:T}, y_{1:T}]$$

$$P_t \equiv P_{t|T} = \text{Cov}(z_t \mid x_{1:T}, y_{1:T})$$

**Cross-covariance** (needed for $A$ update):

$$P_{t, t-1|T} = \text{Cov}(z_t, z_{t-1} \mid x_{1:T}, y_{1:T}) = G_{t-1} P_{t|T}$$

Note: this uses the relation $P_{t,t-1|T} = P_{t|T} G_{t-1}^\top$, so $P_{t-1,t|T} = G_{t-1} P_{t|T}$.

**Aggregated sufficient statistics:**

$$S_{11} = \sum_{t=2}^{T} \left( P_{t-1|T} + \hat{z}_{t-1|T} \hat{z}_{t-1|T}^\top \right) \quad (r \times r)$$

$$S_{10} = \sum_{t=2}^{T} \left( P_{t, t-1|T} + \hat{z}_{t|T} \hat{z}_{t-1|T}^\top \right) \quad (r \times r)$$

$$S_{00} = \sum_{t=2}^{T} \left( P_{t|T} + \hat{z}_{t|T} \hat{z}_{t|T}^\top \right) \quad (r \times r)$$

**For the observation model:**

$$\hat{Z} \in \mathbb{R}^{T \times r}, \quad \text{rows are } \hat{z}_{t|T}^\top$$

$$S_{zz}^{\text{obs}} = \sum_{t=1}^{T} \left( P_{t|T} + \hat{z}_{t|T} \hat{z}_{t|T}^\top \right) \quad (r \times r)$$

### 2.2 M-Step

The complete-data log-likelihood decomposes into three independent terms:

$$\mathcal{L}_c = \underbrace{\log p(z_1)}_{\text{initial}} + \underbrace{\sum_{t=2}^{T} \log p(z_t \mid z_{t-1}, x_t)}_{\text{dynamics}} + \underbrace{\sum_{t=1}^{T} \log p(y_t \mid z_t)}_{\text{observations}}$$

#### 2.2.1 Joint Update for $A$ and $U$ (Dynamics Parameters)

The dynamics term is:

$$Q_{\text{dyn}} = -\frac{T-1}{2} \log|\Psi| - \frac{1}{2} \sum_{t=2}^{T} \mathbb{E}\left[ (z_t - A z_{t-1} - U^\top x_t)^\top \Psi^{-1} (z_t - A z_{t-1} - U^\top x_t) \right]$$

Define the concatenated predictor:

$$f_t = \begin{bmatrix} z_{t-1} \\ x_t \end{bmatrix} \in \mathbb{R}^{r + m}, \quad \Theta = \begin{bmatrix} A^\top \\ U \end{bmatrix} \in \mathbb{R}^{(r+m) \times r}$$

so that $A z_{t-1} + U^\top x_t = \Theta^\top f_t$.

The joint update is expected least squares:

$$\Theta^{\text{new}} = \left( \sum_{t=2}^{T} \mathbb{E}[f_t f_t^\top] \right)^{-1} \left( \sum_{t=2}^{T} \mathbb{E}[f_t z_t^\top] \right)$$

where:

$$\sum_{t=2}^{T} \mathbb{E}[f_t f_t^\top] = \begin{bmatrix} S_{11} & \sum_t \hat{z}_{t-1} x_t^\top \\ \sum_t x_t \hat{z}_{t-1}^\top & \sum_t x_t x_t^\top \end{bmatrix}$$

$$\sum_{t=2}^{T} \mathbb{E}[f_t z_t^\top] = \begin{bmatrix} S_{10}^\top \\ \sum_t x_t \hat{z}_t^\top \end{bmatrix}$$

Note: $S_{10}^\top$ appears here because $S_{10} = \sum \mathbb{E}[z_t z_{t-1}^\top]$, so $\sum \mathbb{E}[z_{t-1} z_t^\top] = S_{10}^\top$.

After solving, extract:

$$A^{\text{new}} = \Theta^{\text{new}}[0{:}r, :]^\top, \quad U^{\text{new}} = \Theta^{\text{new}}[r{:}, :]$$

#### 2.2.2 Update for $\Psi$ (Process Noise)

Using the new $\Theta^{\text{new}}$:

$$\Psi^{\text{new}} = \frac{1}{T-1} \left[ S_{00} - \Theta^{\text{new}\top} \left(\sum_{t=2}^T \mathbb{E}[f_t z_t^\top]\right) - \left(\sum_{t=2}^T \mathbb{E}[z_t f_t^\top]\right) \Theta^{\text{new}} + \Theta^{\text{new}\top} \left(\sum_{t=2}^T \mathbb{E}[f_t f_t^\top]\right) \Theta^{\text{new}} \right]$$

Or more compactly, since $\Theta^{\text{new}}$ is the least-squares solution:

$$\Psi^{\text{new}} = \frac{1}{T-1} \left[ S_{00} - \left(\sum_{t=2}^T \mathbb{E}[f_t z_t^\top]\right)^\top \Theta^{\text{new}} \right]$$

If isotropic: $\psi^2 = \text{Tr}(\Psi^{\text{new}}) / r$. If diagonal: take $\text{diag}(\Psi^{\text{new}})$.

#### 2.2.3 Update for $V$ (Observation Decoder)

$$V^{\text{new}} = Y^\top \hat{Z} \cdot \left(S_{zz}^{\text{obs}}\right)^{-1}$$

#### 2.2.4 Update for $\Sigma$ (Observation Noise)

$$\Sigma^{\text{new}} = \frac{1}{T} \left[ (Y - \hat{Z} V^{\text{new}\top})^\top (Y - \hat{Z} V^{\text{new}\top}) + V^{\text{new}} \left(\sum_{t=1}^T P_{t|T}\right) V^{\text{new}\top} \right]$$

Same structure as static PRRR: empirical residuals + propagated posterior uncertainty.

If isotropic: $\sigma^2 = \text{Tr}(\Sigma^{\text{new}}) / n$. If diagonal: take $\text{diag}(\Sigma^{\text{new}})$.

#### 2.2.5 Update for Initial State

$$\mu_0^{\text{new}} = \hat{z}_{1|T}$$

$$\Sigma_0^{\text{new}} = P_{1|T}$$

### 2.3 Log-Likelihood for Monitoring

The marginal log-likelihood $\log p(y_{1:T} \mid x_{1:T}, \theta)$ is computed as a byproduct of the Kalman filter forward pass via the innovation decomposition:

$$\log p(y_{1:T} \mid x_{1:T}, \theta) = \sum_{t=1}^{T} \log p(y_t \mid y_{1:t-1}, x_{1:t}, \theta)$$

where each term uses the innovation:

$$e_t = y_t - V \hat{z}_{t|t-1}$$

$$S_t = V P_{t|t-1} V^\top + \Sigma$$

$$\log p(y_t \mid y_{1:t-1}, x_{1:t}) = -\frac{n}{2} \log(2\pi) - \frac{1}{2} \log|S_t| - \frac{1}{2} e_t^\top S_t^{-1} e_t$$

This should monotonically increase across EM iterations.

---

## 3. Prediction and Uncertainty Quantification

### 3.1 Filtered Prediction (Online)

At time $t$, using only past and present observations:

$$\hat{y}_{t|t} = V \hat{z}_{t|t}$$

This uses the Kalman filter output and is available in real time.

### 3.2 Smoothed Prediction (Offline)

Using all observations:

$$\hat{y}_{t|T} = V \hat{z}_{t|T}$$

This uses the RTS smoother output and gives better estimates but requires the full sequence.

### 3.3 One-Step-Ahead Prediction

Predict the next output given history:

$$\hat{y}_{t+1|t} = V \hat{z}_{t+1|t} = V(A \hat{z}_{t|t} + U^\top x_{t+1})$$

$$\text{Cov}(y_{t+1} \mid y_{1:t}, x_{1:t+1}) = V P_{t+1|t} V^\top + \Sigma = V(A P_{t|t} A^\top + \Psi) V^\top + \Sigma$$

### 3.4 Multi-Step-Ahead Prediction

For $k$-step-ahead prediction, recursively apply:

$$\hat{z}_{t+k|t} = A \hat{z}_{t+k-1|t} + U^\top x_{t+k}$$

$$P_{t+k|t} = A P_{t+k-1|t} A^\top + \Psi$$

$$\hat{y}_{t+k|t} = V \hat{z}_{t+k|t}$$

Note: This requires knowledge of future inputs $x_{t+1}, \ldots, x_{t+k}$. If those are unavailable, set them to zero for autonomous prediction, or model $x_t$ separately.

### 3.5 Scoring

Use $R^2$ on one-step-ahead predictions for comparison with standard RRR:

$$R^2 = 1 - \frac{\sum_t \|y_t - \hat{y}_{t|t-1}\|^2}{\sum_t \|y_t - \bar{y}\|^2}$$

Also report the marginal log-likelihood from the Kalman filter (§2.3) for model comparison across ranks and for comparing against static PRRR.

---
