import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Set random seed for reproducibility
np.random.seed(42)

# True observation value
y_true = 5.0

# Predicted distribution parameters (e.g., from a forecast model)
predicted_mean = 4.5
predicted_std = 1.2

# Create x-axis range for plotting CDFs
x_min = min(y_true - 3 * predicted_std, predicted_mean - 4 * predicted_std)
x_max = max(y_true + 3 * predicted_std, predicted_mean + 4 * predicted_std)
x = np.linspace(x_min, x_max, 1000)

# Ideal CDF: step function at y_true
ideal_cdf = np.where(x < y_true, 0.0, 1.0)

# Predicted CDF: normal distribution CDF
predicted_cdf = stats.norm.cdf(x, loc=predicted_mean, scale=predicted_std)

# Compute CRPS manually for verification
# CRPS = ∫ [F_predicted(x) - 1(x ≥ y_true)]² dx
squared_diff = (predicted_cdf - ideal_cdf) ** 2
# Approximate integral using trapezoidal rule
crps_value = np.trapz(squared_diff, x)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Color scheme matching the course theme
color_ideal = (135/255, 108/255, 173/255)  # Primary purple
color_predicted = (106/255, 177/255, 208/255)  # Light blue
color_area = (0.7, 0.7, 0.7)  # Light red for the area

# Plot the area between the two CDFs (shaded region)
# The area represents the squared difference integrated over x
# Fill between the two CDFs to visualize CRPS
ax.fill_between(x, ideal_cdf, predicted_cdf, 
                alpha=0.3, color=color_area, 
                label=f'CRPS area', zorder=1)

# Plot ideal CDF (step function)
ax.plot(x, ideal_cdf, color=color_ideal, linewidth=2.5,
        label='Ideal CDF', zorder=3)

# Plot predicted CDF (smooth curve)
ax.plot(x, predicted_cdf, color=color_predicted, linewidth=2.5,
        linestyle='-', label='Predicted CDF', zorder=2)

# Add vertical line at the true observation
ax.axvline(x=y_true, color=color_ideal, linestyle='--', linewidth=1.5,
           alpha=0.6, zorder=4)

# Formatting
ax.set_xlabel('Value', fontsize=20)
ax.set_ylabel('Cumulative Probability', fontsize=20)
ax.legend(loc='best', fontsize=20)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(x_min, x_max)

plt.tight_layout()
plt.savefig('slides/fig/crps.svg', bbox_inches='tight')
plt.close()
