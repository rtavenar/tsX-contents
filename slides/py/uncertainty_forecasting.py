import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Generate synthetic time series data
np.random.seed(42)
n_past = 100
n_future = 20
t_past = np.arange(n_past)
t_future = np.arange(n_past, n_past + n_future)

# Generate past observations with trend and seasonality
trend = 0.01 * t_past
seasonal = 2 * np.sin(2 * np.pi * t_past / 20)
noise = np.random.randn(n_past) * 0.5
past_values = 10 + trend + seasonal + noise

# Generate ground truth future (for comparison)
trend_future = 0.01 * t_future
seasonal_future = 2 * np.sin(2 * np.pi * t_future / 20)
noise_future = np.random.randn(n_future) * 0.5
future_true = 10 + trend_future + seasonal_future + noise_future

# Simulate forecast predictions (point estimates)
# In practice, these would come from a trained model
forecast_mean = 10 + 0.01 * t_future + 2 * np.sin(2 * np.pi * t_future / 20)

# Simulate uncertainty that grows over the forecast horizon
# This is typical: uncertainty increases as we forecast further ahead
base_uncertainty = 0.5
uncertainty_growth = np.linspace(0.5, 2.0, n_future)
forecast_std = base_uncertainty + uncertainty_growth

# Compute quantiles for uncertainty bands
# 50% interval (interquartile range)
q25 = forecast_mean - 0.6745 * forecast_std  # ~25th percentile for normal
q75 = forecast_mean + 0.6745 * forecast_std  # ~75th percentile for normal

# 90% interval
q05 = forecast_mean - 1.645 * forecast_std  # ~5th percentile
q95 = forecast_mean + 1.645 * forecast_std  # ~95th percentile

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Color scheme matching the course theme
color_past = (135/255, 108/255, 173/255)  # Primary purple
color_forecast = (106/255, 177/255, 208/255)  # Light blue
color_true = (0.2, 0.6, 0.2)  # Green for ground truth

# Plot past observations
ax.plot(t_past, past_values, color=color_past, linewidth=2, 
        label='Past observations', zorder=3)

# Plot uncertainty bands (90% interval)
ax.fill_between(t_future, q05, q95, alpha=0.2, color=color_forecast,
                label='90% prediction interval', zorder=1)

# Plot uncertainty bands (50% interval)
ax.fill_between(t_future, q25, q75, alpha=0.3, color=color_forecast,
                label='50% prediction interval', zorder=2)

# Plot point forecast
ax.plot(t_future, forecast_mean, color=color_forecast, linewidth=2,
        linestyle='--', marker='o', markersize=4, 
        label='Point forecast', zorder=4)

# Plot ground truth (if available)
ax.plot(t_future, future_true, color=color_true, linewidth=2,
        linestyle='-', marker='s', markersize=4,
        label='Ground truth', zorder=5)

# Add vertical line to separate past and future
ax.axvline(x=n_past-0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Formatting
# ax.set_xlabel('Time', fontsize=12)
# ax.set_ylabel('Value', fontsize=12)
# ax.set_title('Multi-step Ahead Forecasting with Uncertainty', fontsize=14, fontweight='bold')
# ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Add text annotation for forecast horizon
ylim = ax.get_ylim()
ax.set_yticklabels([])
ax.set_xticklabels([])
# ax.text(n_past + n_future/2, ylim[1] * 0.95, 
#         'Forecast horizon', ha='center', fontsize=10, 
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('slides/fig/uncertainty_forecasting.svg', bbox_inches='tight')
plt.close()
