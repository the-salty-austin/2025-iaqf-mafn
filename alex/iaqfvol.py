import numpy as np
import matplotlib.pyplot as plt

# -- Parameters --
N = 100             # Number of stocks
num_sims = 10000    # Number of simulations

# Beta distribution parameters
mean_beta = 1.0     # Mean of beta distribution
std_beta = 0.3      # Std. dev. of beta distribution

# Factor & residual std dev
sigma_factor = 0.15  # Std dev of the factor (market)
sigma_resid  = 0.3  # Std dev of the residuals

# Residual correlation
rho = 0          # Correlation among all residuals

# Build the NxN correlation matrix for residuals
# Diagonal = 1.0, off-diagonal = rho
corr_matrix = np.full((N, N), rho)
np.fill_diagonal(corr_matrix, 1.0)

# We'll store results here
hhi_vals = []
port_std_vals = []

for _ in range(num_sims):
    # 1) Generate random weights that sum to 1
    w = np.random.rand(N)
    w = w / w.sum()  # normalize so sum(w) = 1

    # 2) Compute HHI = sum of squared weights
    hhi = np.sum(w**2)

    # 3) Generate random betas from Normal(mean_beta, std_beta)
    betas = np.random.normal(mean_beta, std_beta, N)

    # 4) Residual covariance matrix = sigma_resid^2 * corr_matrix
    resid_cov = (sigma_resid**2) * corr_matrix

    # 5) Factor covariance contribution:
    #    factor_cov = sigma_factor^2 * (betas outer betas)
    factor_cov = sigma_factor**2 * np.outer(betas, betas)

    # 6) Total covariance matrix
    Sigma = factor_cov + resid_cov

    # 7) Compute portfolio variance = w^T * Sigma * w
    port_var = w @ Sigma @ w

    # 8) Portfolio std dev
    port_std = np.sqrt(port_var)

    # Store the results
    hhi_vals.append(hhi)
    port_std_vals.append(port_std)

# Convert to NumPy arrays
hhi_vals      = np.array(hhi_vals)
port_std_vals = np.array(port_std_vals)

# -- Plotting --
plt.figure(figsize=(8,6))
plt.scatter(hhi_vals, port_std_vals, alpha=0.4, s=10)
plt.xlabel("HHI (Herfindahl–Hirschman Index)")
plt.ylabel("Portfolio Standard Deviation")
plt.title("Portfolio Std vs. HHI (Single-Factor, Residual Corr = 0.3)")
plt.grid(True)
plt.show()

# (Optional) Compute the correlation between HHI and the portfolio std dev
corr_coefficient = np.corrcoef(hhi_vals, port_std_vals)[0, 1]
print(f"Correlation between HHI and Portfolio Std Dev: {corr_coefficient:.4f}")
