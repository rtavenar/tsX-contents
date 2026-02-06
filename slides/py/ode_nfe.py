#!/usr/bin/env python3
"""
Euler solver – effect of step count on spiraling trajectories
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non‑interactive backend
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1.  ODE definition – stable focus (spiral towards (0,0))
# ------------------------------------------------------------------
def system(t, y):
    """
    2‑D linear system that produces a decaying spiral.
    y : array_like, shape (2,)
        [x, y] components.
    Returns
    -------
    dydt : ndarray, shape (2,)
    """
    x, y_ = y
    return np.array([-x - y_,      # dx/dt
                     x - y_])     # dy/dt

# ------------------------------------------------------------------
# 2.  Exact solution (for reference)
# ------------------------------------------------------------------
def exact_solution(t, y0):
    """Return the analytic solution of the above system at time t."""
    x0, y0_ = y0
    expm = np.exp(-t)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    x = expm * (x0 * cos_t - y0_ * sin_t)
    y = expm * (x0 * sin_t + y0_ * cos_t)
    return t, np.vstack([x, y]).T   # shape (len(t), 2)

# ------------------------------------------------------------------
# 3.  Euler integrator
# ------------------------------------------------------------------
def euler(f, y0, t0, t_end, N):
    """
    Simple explicit Euler solver.

    Parameters
    ----------
    f   : function(t, y) -> ndarray
    y0  : array_like, shape (2,)
    t0  : float, initial time
    t_end: float, final time
    N   : int, number of steps

    Returns
    -------
    t   : ndarray, shape (N+1,)
    y   : ndarray, shape (N+1, 2)
    """
    h = (t_end - t0) / N
    t = np.linspace(t0, t_end, N + 1)
    y = np.empty((N + 1, 2))
    y[0] = y0
    for i in range(N):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y

# ------------------------------------------------------------------
# 4.  Setup: starting points & colours
# ------------------------------------------------------------------
colors = [
    (135/255, 108/255, 173/255),  # Primary purple
    (106/255, 177/255, 208/255),  # Light blue
    (0.7, 0.7, 0.7)                # Light grey for area (used below)
]

starting_points = [
    (np.cos(np.pi/2), np.sin(np.pi/2)),
    (np.cos(np.pi/2 + 2 * np.pi / 3), np.sin(np.pi/2 + 2 * np.pi / 3)),
    (np.cos(np.pi/2 - 2 * np.pi / 3), np.sin(np.pi/2 - 2 * np.pi / 3)),
]

# ------------------------------------------------------------------
# 5.  Simulation parameters
# ------------------------------------------------------------------
T = 10.0                     # total integration time
step_counts = [20, 30, 50]   # different numbers of steps

# ------------------------------------------------------------------
# 6.  Plotting
# ------------------------------------------------------------------
plt.figure()

# For each starting point, compute Euler trajectory
for (x0, y0), col, N in zip(starting_points, colors[:3], step_counts):
    t_ex, y_ex = exact_solution(np.linspace(0, T, 1000), (x0, y0))
    plt.plot(y_ex[:, 0], y_ex[:, 1], color='black', lw=0.8, ls='--',
            label='Exact solution' if y0 == 1. else None)
    
    t, y = euler(system, np.array([x0, y0]), 0.0, T, N)
    plt.plot(y[:, 0], y[:, 1], color=col, lw=3., marker="o", label=f"{N} steps")

plt.gca().set_aspect('equal')
plt.legend(loc='upper right', fontsize='large')
plt.xticks([])
plt.yticks([])

# Tight layout and save
plt.tight_layout()
plt.savefig('slides/fig/ode_nfe.svg', bbox_inches='tight')
