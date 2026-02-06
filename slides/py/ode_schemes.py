#!/usr/bin/env python3
"""
Runge–Kutta vs. Euler on a decaying spiral ODE.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non‑interactive backend
import matplotlib.pyplot as plt

# Set larger fonts
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# ------------------------------------------------------------------
# 1. ODE definition – stable focus (spiral towards (0,0))
# ------------------------------------------------------------------
def system(t, y):
    """dx/dt = -x - y,   dy/dt =  x - y."""
    x, y_ = y
    return np.array([-x - y_, x - y_])

# ------------------------------------------------------------------
# 2. Exact solution (for reference)
# ------------------------------------------------------------------
def exact_solution(t, y0):
    """Analytic solution of the above system."""
    x0, y0_ = y0
    expm = np.exp(-t)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    x = expm * (x0 * cos_t - y0_ * sin_t)
    y = expm * (x0 * sin_t + y0_ * cos_t)
    return t, np.vstack([x, y]).T

# ------------------------------------------------------------------
# 3. Integrators
# ------------------------------------------------------------------
def euler(f, y0, t0, t_end, N):
    """Explicit Euler."""
    h = (t_end - t0) / N
    t = np.linspace(t0, t_end, N + 1)
    y = np.empty((N + 1, 2))
    y[0] = y0
    for i in range(N):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y

def rk2(f, y0, t0, t_end, N):
    """Heun (improved Euler) – 2‑stage RK."""
    h = (t_end - t0) / N
    t = np.linspace(t0, t_end, N + 1)
    y = np.empty((N + 1, 2))
    y[0] = y0
    for i in range(N):
        k1 = f(t[i],     y[i])
        k2 = f(t[i]+h, y[i] + h*k1)
        y[i+1] = y[i] + h*(k1 + k2)/2
    return t, y

def rk4(f, y0, t0, t_end, N):
    """Classical 4‑stage RK."""
    h = (t_end - t0) / N
    t = np.linspace(t0, t_end, N + 1)
    y = np.empty((N + 1, 2))
    y[0] = y0
    for i in range(N):
        k1 = f(t[i],     y[i])
        k2 = f(t[i]+h/2, y[i] + h*k1/2)
        k3 = f(t[i]+h/2, y[i] + h*k2/2)
        k4 = f(t[i]+h,   y[i] + h*k3)
        y[i+1] = y[i] + h*(k1 + 2*k2 + 2*k3 + k4)/6
    return t, y

# ------------------------------------------------------------------
# 4. Simulation parameters
# ------------------------------------------------------------------
y0 = (np.cos(np.pi/2), np.sin(np.pi/2))   # (0, 1)
T   = 10.0
N   = 20                                  # same step count for all schemes

# List of schemes: (label, function, stages)
schemes = [
    ("Euler (RK1)",   euler, 1),
    ("Heun (RK2)",    rk2,   2),
    ("RK4 (Classical)", rk4, 4),
]

# ------------------------------------------------------------------
# 5. Run the schemes
# ------------------------------------------------------------------
results = {}
for label, func, _ in schemes:
    t, y = func(system, np.array(y0), 0.0, T, N)
    results[label] = (t, y)

# Exact solution
t_ex, y_ex = exact_solution(np.linspace(0, T, 1000), y0)

# ------------------------------------------------------------------
# 6. Plotting
# ------------------------------------------------------------------
plt.figure(figsize=(6, 5))

# Error vs. NFE
ax = plt.gca()
Ns = np.logspace(0.5, 2, 20, dtype=int)   # 3 to 100 steps
errors = {label: [] for label, _, _ in schemes}
nfes = {label: [] for label, _, _ in schemes}

for Nstep in Ns:
    for (label, func, stages) in schemes:
        t, y = func(system, np.array(y0), 0.0, T, Nstep)
        # exact state at final time
        _, y_exact_T = exact_solution(np.array([T]), y0)
        err = np.linalg.norm(y[-1] - y_exact_T[0])
        errors[label].append(err)
        nfes[label].append(Nstep * stages)

colors = [
    (135/255, 108/255, 173/255),  # Primary purple
    (106/255, 177/255, 208/255),  # Light blue
    (212/255, 62/255, 136/255)    # Magenta
]
for (label, _, _), c in zip(schemes, colors):
    ax.semilogy(nfes[label], errors[label], marker='o', linestyle='-', color=c, label=label)
ax.set_xlabel('Number of function evaluations (NFE)')
ax.set_ylabel('Euclidean error')
# ax.set_title('Error vs. NFE')
ax.grid(True, which='both', ls=':')
ax.set_xlim([0, 100])
ax.legend()

plt.tight_layout()
plt.savefig('slides/fig/ode_rk.svg', bbox_inches='tight')
