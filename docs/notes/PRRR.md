# Standard RRR
$$
y_t = W^\top x_t + \epsilon_t \quad W=VU^\top \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2, I_n)
$$
# Latent Communication PRRR

$$
\mathbf{z} = \mathbf{U^\top x} + \boldsymbol{\eta} \quad \boldsymbol{\eta}\sim\boldsymbol{\mathcal{N}}(\mathbf{0}, \mathbb{\Psi})
$$
$$
\mathbf{y} = \mathbf{Vz} + \boldsymbol{\epsilon} \quad \boldsymbol{\epsilon}\sim \boldsymbol{\mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma})}
$$
$$
\begin{aligned}
\mathbf{y} &= \mathbf{V}(\mathbf{U}^\top\mathbf{x} + \boldsymbol{\eta}) + \boldsymbol{\epsilon}
\\
&=\mathbf{VU^\top x} + \mathbf{V}\boldsymbol{\eta} + \boldsymbol{\epsilon}
\end{aligned}
$$
## Joint distribution
So the complete probability distribution

$$
\begin{bmatrix}
\mathbf{z} \\ \mathbf{y}
\end{bmatrix}
\sim
\mathcal{N}
\bigg(
\begin{bmatrix}
\mathbb{E}[\mathbf{z}] \\
\mathbb{E}[\mathbf{y}]
\end{bmatrix},
\begin{bmatrix}
\mathbb{V}[\mathbf{z}] & \text{cov}(\mathbf{z}, \mathbf{y}) \\
\text{cov}(\mathbf{y}, \mathbf{z}) & \mathbb{V}[\mathbf{y}]


\end{bmatrix}
\bigg)
$$
### Means
$$
\mathbb{E}[\mathbf{z}|\mathbf{x}] = \mathbf{U}^\top\mathbf{x}
\quad
\mathbb{E}[\mathbf{y}|\mathbf{x}] = \mathbf{VU}^\top\mathbf{x}
$$
### Variances
$$
\mathbb{V}[\mathbf{z}|\mathbf{x}] = \Psi
$$
$$
\begin{aligned}
\mathbb{V}[\mathbf{y} | \mathbf{x}]
&= \mathbb{E}[\mathbb{V}[\mathbf{y}|\mathbf{z}]|\mathbf{x}] + \mathbb{V}[\mathbb{E}[\mathbf{y}|\mathbf{z}]|\mathbf{x}]
\\
&= \mathbb{E}[\Sigma | \mathbf{x}] + \mathbb{V}[\mathbf{Vz}|\mathbf{x}]
\\
&= \Sigma +\mathbf{V}\mathbb{V}[\mathbf{z}|\mathbf{x}]\mathbf{V}^\top
\\
&= \mathbf{V}\boldsymbol\Psi\mathbf{V}^\top + \boldsymbol{\Sigma}
\end{aligned} 
$$
### Cross-Covariances
$$
\mathbf{cov}[\mathbf{z}, \mathbf{y}] = \mathbb{E}\bigg[(\mathbf{z} - \mathbb{E}[\mathbf{z}])(\mathbf{y} - \mathbb{E}[\mathbf{y}])^\top\bigg]
$$
$$
\mathbf{z} - \mathbb{E}[\mathbf{z}] = (\mathbf{U^\top x} + \boldsymbol{\eta}) - \mathbf{U^\top x} = \boldsymbol{\eta}
$$
$$
\mathbf{y} - \mathbb{E}[\mathbf{y}] = (\mathbf{VU^\top x} + \mathbf{V}\boldsymbol{\eta} + \boldsymbol{\epsilon}) - \mathbf{VU^\top x} = \mathbf{V}\boldsymbol{\eta} + \boldsymbol{\epsilon}
$$
$$
\begin{aligned}
\mathbf{cov}[\mathbf{z}, \mathbf{y}] &=
\mathbb{E}\bigg[(\mathbf{z} - \mathbb{E}[\mathbf{z}])(\mathbf{y} - \mathbb{E}[\mathbf{y}])^\top\bigg]
\\
&= \mathbb{E}[\boldsymbol{\eta}(\mathbf{V}\boldsymbol{\eta} + \boldsymbol{\epsilon})^\top]
\\
&=\mathbb{E}[\boldsymbol{\eta\eta}^\top\mathbf{V}^\top] + \mathbb{E}[\boldsymbol{\eta\boldsymbol{\epsilon}}^\top]
\\
&=\mathbb{E}[\boldsymbol{\eta\eta}^\top]\mathbf{V^\top}
\\
&=\boldsymbol\Psi \mathbf{V}^\top
\end{aligned}
$$
$$
\mathbf{cov}[\mathbf{y}, \mathbf{z}] = \mathbf{V}\boldsymbol{\Psi}
$$
### Posterior Distribution
$$
p(\mathbf{z}|\mathbf{y}, \boldsymbol{\theta}) = \boldsymbol{\mathcal{N}}(\boldsymbol\gamma_{\mathbf{z}|\mathbf{y}}, \boldsymbol{\Phi}_{\mathbf{z}|\mathbf{y}})
$$
$$
\boldsymbol\gamma_{\mathbf{z}|\mathbf{y}} =
\mathbf{U}^\top\mathbf{x} + (\boldsymbol{\Psi}\mathbf{V}^\top)(\mathbf{V}\boldsymbol{\Psi}\mathbf{V}^\top + \boldsymbol{\Sigma})^{-1}(\mathbf{y} - \mathbf{VU^\top x})
$$
$$
\boldsymbol{\Phi}_{\mathbf{z}|\mathbf{y}} = 
\boldsymbol{\Psi} - (\boldsymbol{\Psi}\mathbf{V}^\top)(\mathbf{V}\boldsymbol{\Psi}\mathbf{V}^\top + \boldsymbol{\Sigma})^{-1}(\mathbf{V}\boldsymbol{\Psi})
$$
Now what we want is to use this distribution to build $\mathcal{Q}(\theta^{(t)} | \theta^{(t - 1)})$
$$
\mathcal{Q}
$$