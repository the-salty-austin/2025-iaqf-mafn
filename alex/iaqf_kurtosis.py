import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Model & Simulation Params
# ----------------------------

N = 500
num_portfolios = 1000

# Factor model params
beta = 0.8            # All stocks have the same beta
sigma_F = 1.0         # Std dev of the factor
sigma_e = 1.0         # Std dev of each residual
kappa_e = 6.0         # Raw kurtosis of each residual (e.g., normal=3, so 5 means heavier tails)

# ----------------------------
# 2) Function to compute raw kurtosis analytically
#    for R_p = beta*F + sum_i w_i e_i
# ----------------------------

def portfolio_kurtosis(w):
    """
    w: array of shape (N,) with sum(w)=1
    Returns: raw kurtosis of portfolio returns, using the formula.
    """
    # a = sum of w_i^2, b = sum of w_i^4
    a = np.sum(w**2)
    b = np.sum(w**4)
    
    # Var(R_p) = beta^2 * sigma_F^2 + sigma_e^2 * a
    var_rp = beta**2 * sigma_F**2 + sigma_e**2 * a
    
    # Fourth moment of R_p:
    # E[R_p^4] = 3 beta^4 sigma_F^4
    #           + 6 beta^2 sigma_F^2 sigma_e^2 a
    #           + sigma_e^4 [3 a^2 + (kappa_e - 3)* b]
    e_rp_4 = (3*beta**4 * sigma_F**4
              + 6*beta**2 * sigma_F**2 * sigma_e**2 * a
              + sigma_e**4 * (3*a**2 + (kappa_e - 3)*b))
    
    # Raw Kurtosis = E[R_p^4] / (Var(R_p))^2
    kurt_rp = e_rp_4 / (var_rp**2)
    
    return kurt_rp

# ----------------------------
# 3) Main Loop over Random Portfolios
# ----------------------------

hhi_vals = []
kurt_vals = []

for _ in range(num_portfolios):
    # Generate random weights that sum to 1
    w = np.random.rand(N)
    w /= w.sum()  # so sum(w)=1
    
    # HHI = sum of w_i^2
    hhi = np.sum(w**2)
    
    # Analytical portfolio kurtosis
    p_kurt = portfolio_kurtosis(w)
    
    # Store
    hhi_vals.append(hhi)
    kurt_vals.append(p_kurt)

# Convert to arrays
hhi_vals = np.array(hhi_vals)
kurt_vals = np.array(kurt_vals)

# ----------------------------
# 4) Plot Kurtosis vs. HHI
# ----------------------------
plt.figure(figsize=(7,5))
plt.scatter(hhi_vals, kurt_vals, alpha=0.5)
plt.xlabel("HHI (Herfindahl–Hirschman Index) = sum of w_i^2")
plt.ylabel("Portfolio Kurtosis (raw)")
plt.title("Analytical Portfolio Kurtosis vs. HHI (10 Stocks, Heavy-Tail Residuals)")
plt.grid(True)
plt.show()

# Optional: Check correlation
corr_val = np.corrcoef(hhi_vals, kurt_vals)[0,1]
print(f"Correlation between HHI and portfolio kurtosis = {corr_val:.3f}")
kurt_vals
