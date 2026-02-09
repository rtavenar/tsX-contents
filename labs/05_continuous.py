# %% [markdown]
# # Continuous-Time Models
#
# In this lab, you will explore continuous-time neural models for trajectory prediction:
# - Neural ODEs: Use learned vector fields to model system dynamics, with automatic differentiation-based solving
# - INRs (Implicit Neural Representations): Directly learn the trajectory mapping from time to position
# - Latent representation adaptation: Optimize latent vectors to fit specific trajectories
#
# In this lab, you will need the `torchdiffeq` package that offers ODE solvers in torch. 
# Install it using the command below and check the docs for its `odeint` function 
# [there](https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/odeint.py).

# %%
%pip install torchdiffeq

# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

torch.manual_seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## Part 0: Setup and Data Generation
#
# **Question 0.** We create a toy dataset of 2D spiral trajectories with noise.
# The trajectories follow a damped rotation dynamics. Understand the `TrajectoryDataset`
# class which will provide initial conditions and ground-truth trajectories for training.

# %%
def generate_trajectory(x0, t, system="spiral", noise_std=0.05):
    x = np.zeros((len(t), 2))
    x[0] = x0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        x_prev = x[i - 1]
        if system == "spiral":
            dx = np.array([-x_prev[1], x_prev[0]]) - 0.1 * x_prev
        else:
            dx = np.array([-x_prev[1], x_prev[0]])
        x[i] = x_prev + dt * dx
    noise = np.random.normal(0, noise_std, x.shape)
    return x + noise

class TrajectoryDataset(Dataset):
    def __init__(self, n_traj=200, t_max=10.0, n_steps=100, system="spiral", noise_std=0.05):
        self.t = np.linspace(0, t_max, n_steps)
        self.data = []
        for _ in range(n_traj):
            x0 = np.random.uniform(-2, 2, size=2)
            traj = generate_trajectory(x0, self.t, system, noise_std=noise_std)
            self.data.append(traj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = self.data[idx]
        return (torch.tensor(self.t, dtype=torch.float32),
                torch.tensor(traj, dtype=torch.float32))

# %%
dataset = TrajectoryDataset(system="spiral")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# %% [markdown]
# ## Part 1: Neural ODEs
#
# **Question 1.** Implement a Neural ODE model that learns the vector field dynamics.
# The model should use `odeint` to solve the ODE from an initial condition across a time grid.
# Train the model on the trajectory dataset and report the training loss.

# %%
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # TODO: define your own `self.net` here to parametrize your ODE

    def forward(self, t, x):
        return self.net(x)

class NeuralODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = ODEFunc()

    def forward(self, t, z):
        # Input z: (B, D), t: (T,) -> Output (B, T, 2)
        # TODO: solve ODE defined by self.func starting from z over time grid t
        raise NotImplementedError()

# %% + tags=["solution"]
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, t, x):
        return self.net(x)

class NeuralODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = ODEFunc()

    def forward(self, t, z):
        # Input z: (B, D), t: (T,) -> Output (B, T, 2)
        return odeint(self.func, z, t).permute(1, 0, 2)

# %%
neural_ode = NeuralODE().to(device)
optimizer_ode = optim.Adam(neural_ode.parameters(), lr=1e-3)

print("--- Training Neural ODE ---")
for epoch in range(101):
    total_loss = 0
    for t, traj in loader:
        t_grid, traj = t[0].to(device), traj.to(device)
        x0 = traj[:, 0] # Initial condition

        pred = neural_ode(t_grid, x0)
        loss = ((pred - traj)**2).mean()

        optimizer_ode.zero_grad()
        loss.backward()
        optimizer_ode.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")
# %% [markdown]
# ## Part 2: Implicit Neural Representations (INRs)
#
# **Question 2.** Implement a conditioned INR that takes an initial position as "latent"
# representation `z`
# and uses modulation to generate trajectory predictions. This model directly learns
# the mapping from time to position, conditioned on the initial state.
# Train it on the trajectory dataset and compare its loss curve with the Neural ODE.

# %%
# Implement the `forward` method below to apply modulation and produce (B, T, 2) outputs.
class ModulatedINR(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.modulation = nn.Linear(latent_dim, 64 * 4) 
        self.input_layer = nn.Linear(1, 64)
        self.hidden_layer = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 2)
        self.activation = nn.ReLU()

    def forward(self, t, z):
        if t.dim() == 1: 
            t = t.unsqueeze(-1)
        B, T = z.shape[0], t.shape[0]
        params = self.modulation(z)
        g1, b1, g2, b2 = torch.split(params, 64, dim=-1)
        t = t.unsqueeze(0).expand(B, T, 1)

        # TODO: implement forward using modulation parameters


# %% + tags=["solution"]
class ModulatedINR(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.modulation = nn.Linear(latent_dim, 64 * 4) 
        self.input_layer = nn.Linear(1, 64)
        self.hidden_layer = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 2)
        self.activation = nn.ReLU()

    def forward(self, t, z):
        if t.dim() == 1: 
            t = t.unsqueeze(-1)
        B, T = z.shape[0], t.shape[0]
        params = self.modulation(z)
        g1, b1, g2, b2 = torch.split(params, 64, dim=-1)
        
        h = t.unsqueeze(0).expand(B, T, 1)
        h = self.activation(g1.unsqueeze(1) * self.input_layer(h) + b1.unsqueeze(1))
        h = self.activation(g2.unsqueeze(1) * self.hidden_layer(h) + b2.unsqueeze(1))
        return self.output_layer(h)


# %%
inr_model = ModulatedINR().to(device)
optimizer_inr = optim.Adam(inr_model.parameters(), lr=1e-3)

print("\n--- Training Conditioned INR ---")
for epoch in range(201):
    total_loss = 0
    for t, traj in loader:
        t_grid, traj = t[0].to(device), traj.to(device)
        x0 = traj[:, 0] # Initial condition

        pred = inr_model(t_grid, x0)
        loss = ((pred - traj)**2).mean()

        optimizer_inr.zero_grad()
        loss.backward()
        optimizer_inr.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")


# %% [markdown]
# ## Part 4: Comparative Analysis and Visualization
#
# **Question 4.** What does the `optimize_latent` function below do? Why do we have to freeze the model weights during this step?

# %%
def optimize_latent(model, t, traj, latent_dim=2, n_steps=150, lr=1e-1):
    """Optimize latent vectors `z` to fit `traj` under `model`.

    Args:
        model: INR model that expects (t, z) and returns (B, T, 2)
        t: torch tensor time grid, shape (T,) or (T,1)
        traj: torch tensor ground-truth trajectory, shape (B, T, 2) or (T,2)
        latent_dim: dimension of z
        n_steps: inner optimization steps
        lr: inner optimizer lr

    Returns:
        z_opt: optimized latent(s), detached, shape (B, latent_dim)
    """
    model.eval()
    # Ensure batch dimension
    if traj.dim() == 2:
        traj = traj.unsqueeze(0)  # (1, T, 2)
    B = traj.shape[0]

    # Freeze model parameters during inner optimization so only z receives gradients
    param_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad = False

    z = torch.randn(B, latent_dim, device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=lr)
    for _ in range(n_steps):
        pred = model(t, z)  # (B, T, 2)
        loss = ((pred - traj)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # restore model parameter flags
    for p, flag in zip(model.parameters(), param_flags):
        p.requires_grad = flag

    return z.detach()


# %% [markdown]
# **Question 5.** Visualize predictions from Neural ODE, conditioned INR, and optimized INR
# on unseen trajectories. For each initial condition:
# - Show training-time predictions (interpolation)
# - Extrapolate beyond the training time window
# - Compare generalization behavior and model differences
#
# Discuss observations: Which model generalizes better? How does test-time
# adaptation with latent optimization compare to the conditioned approaches?

# %%
T_train = dataset.t[-1]
t_test = torch.linspace(0.0, 1.5 * T_train, 200).to(device)
indices_interp = (t_test <= T_train).cpu()
indices_extrap = (t_test > T_train).cpu()

@torch.no_grad()
def rollout(model, z, t):
    """Run model rollout given either an unbatched initial code `z` (shape (D,))
    or a batched latent `z` (shape (B, D)). Returns the first batch element CPU tensor.
    """
    model.eval()
    if z.dim() == 1:
        z = z.unsqueeze(0)
    return model(t, z)[0].cpu()

# Pick new unseen initial conditions
x0_tests = torch.tensor(np.random.uniform(-2.5, 2.5, size=(2, 2)), dtype=torch.float32).to(device)

plt.figure(figsize=(12, 6))

for i, x0 in enumerate(x0_tests):
    # GT
    gt = generate_trajectory(x0.cpu().numpy(), t_test.cpu().numpy(), noise_std=0.)
    t_train_tensor = torch.tensor(dataset.t, dtype=torch.float32).to(device)
    gt_train_part = torch.tensor(generate_trajectory(x0.cpu().numpy(), dataset.t), dtype=torch.float32).to(device)
    y_samples = gt_train_part
    
    # Model 1: Neural ODE (Generalizes by vector field)
    p_node = rollout(neural_ode, x0, t_test)
    
    # Model 2: Conditioned INR (Generalizes by x0-modulation)
    p_inr_cond = rollout(inr_model, x0, t_test)

    # Model 3: INR (Test-time optimized latent for conditioned INR)
    # we optimize a latent for the conditioned INR model at test time
    z_opt = optimize_latent(inr_model, t_train_tensor, gt_train_part)
    p_inr_opt = inr_model(t_test, z_opt)[0].detach().cpu()

    plt.subplot(1, 2, i + 1)
    plt.plot(gt[indices_interp, 0], gt[indices_interp, 1], "k-", label="GT")
    plt.plot(gt[indices_extrap, 0], gt[indices_extrap, 1], "k--", alpha=0.5, label="GT (Extrap)")
    plt.scatter(y_samples[:, 0], y_samples[:, 1], c='gray', alpha=0.5, s=15, label="Noisy Samples")
    plt.plot(p_node[:, 0], p_node[:, 1], color="blue", label="Neural ODE")
    plt.plot(p_inr_cond[:, 0], p_inr_cond[:, 1], color="orange", label="INR (Fixed x0)")
    plt.plot(p_inr_opt[:, 0], p_inr_opt[:, 1], color="green", ls=":", label="INR (Optimized z)")
    plt.scatter(x0[0].cpu(), x0[1].cpu(), c="red", s=30)
    plt.axis("equal")
    plt.title(f"Test IC {i}")
    if i == 0: 
        plt.legend(fontsize=7)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Question 6.** Use SIREN activations in your Modulated INR model above. 
# Visualize the impact it has on predictions (using the same setup as above).

# %%
# Implement the `forward` method to use sinusoidal activations (SIREN-like).
class ModulatedINR(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        # TODO: edit the __init__ from previous implementation

    def forward(self, t, z):
        # TODO: implement forward using sine activations
        raise NotImplementedError()

# %% + tags=["solution"]
class ModulatedINR(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.modulation = nn.Linear(latent_dim, 64 * 4) 
        self.input_layer = nn.Linear(1, 64)
        self.hidden_layer = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 2)
        self.activation = torch.sin

    def forward(self, t, z):
        if t.dim() == 1: 
            t = t.unsqueeze(-1)
        B, T = z.shape[0], t.shape[0]
        params = self.modulation(z)
        g1, b1, g2, b2 = torch.split(params, 64, dim=-1)
        
        h = t.unsqueeze(0).expand(B, T, 1)
        h = self.activation(g1.unsqueeze(1) * self.input_layer(h) + b1.unsqueeze(1))
        h = self.activation(g2.unsqueeze(1) * self.hidden_layer(h) + b2.unsqueeze(1))

        return self.output_layer(h)

# %%
inr_model = ModulatedINR().to(device)
optimizer_inr = optim.Adam(inr_model.parameters(), lr=1e-3)

print("\n--- Training Conditioned INR ---")
for epoch in range(201):
    total_loss = 0
    for t, traj in loader:
        t_grid, traj = t[0].to(device), traj.to(device)
        x0 = traj[:, 0] # Initial condition

        pred = inr_model(t_grid, x0)
        loss = ((pred - traj)**2).mean()

        optimizer_inr.zero_grad()
        loss.backward()
        optimizer_inr.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

# %%
T_train = dataset.t[-1]
t_test = torch.linspace(0.0, 1.5 * T_train, 200).to(device)
indices_interp = (t_test <= T_train).cpu()
indices_extrap = (t_test > T_train).cpu()

@torch.no_grad()
def rollout(model, x0, t):
    model.eval()
    return model(t, x0.unsqueeze(0))[0].cpu()

# Pick new unseen initial conditions
x0_tests = torch.tensor(np.random.uniform(-2.5, 2.5, size=(2, 2)), dtype=torch.float32).to(device)

plt.figure(figsize=(12, 6))

for i, x0 in enumerate(x0_tests):
    # GT
    gt = generate_trajectory(x0.cpu().numpy(), t_test.cpu().numpy(), noise_std=0.)
    t_train_tensor = torch.tensor(dataset.t, dtype=torch.float32).to(device)
    gt_train_part = torch.tensor(generate_trajectory(x0.cpu().numpy(), dataset.t), dtype=torch.float32).to(device)
    y_samples = gt_train_part
    
    # Model 1: Neural ODE (Generalizes by vector field)
    p_node = rollout(neural_ode, x0, t_test)
    
    # Model 2: Conditioned INR (Generalizes by x0-modulation)
    p_inr_cond = rollout(inr_model, x0, t_test)

    plt.subplot(1, 2, i + 1)
    plt.plot(gt[indices_interp, 0], gt[indices_interp, 1], "k-", label="GT")
    plt.plot(gt[indices_extrap, 0], gt[indices_extrap, 1], "k--", alpha=0.5, label="GT (Extrap)")
    plt.scatter(y_samples[:, 0], y_samples[:, 1], c='gray', alpha=0.5, s=15, label="Noisy Samples")
    plt.plot(p_node[:, 0], p_node[:, 1], color="blue", label="Neural ODE")
    plt.plot(p_inr_cond[:, 0], p_inr_cond[:, 1], color="orange", label="INR (Fixed x0)")
    
    plt.scatter(x0[0].cpu(), x0[1].cpu(), c="red", s=30)
    plt.axis("equal")
    plt.title(f"Test IC {i}")
    if i == 0: plt.legend(fontsize=7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Bonus: Train-time latent optimization
#
# This optional section shows how to train an INR when latents are not provided
# by `x0` but are instead *obtained during training* by optimizing a latent
# per batch (inner-loop). The procedure is:
# - For each batch, run an inner optimization to find `z` that fits the batch
#   under the current model (model parameters frozen during inner loop).
# - Use the optimized `z` as fixed targets to update the INR parameters
#   (outer loop).
#
# **Question.** Update the training loop used above to match this training procedure.

# %% + tags=["solution"]
latent_dim = 8
inr_latent = ModulatedINR(latent_dim=latent_dim).to(device)
optimizer_inr_latent = optim.Adam(inr_latent.parameters(), lr=1e-3)

print("\n--- Training INR with train-time latent optimization ---")
for epoch in range(201):
    total_loss = 0
    for t, traj in loader:
        t_grid, traj = t[0].to(device), traj.to(device)

        # Inner loop: find z that reconstructs traj under current model
        z_opt = optimize_latent(inr_latent, t_grid, traj, latent_dim=latent_dim, n_steps=60, lr=1e-1)

        # Outer loop: update model parameters using the optimized latents
        pred = inr_latent(t_grid, z_opt)
        loss = ((pred - traj)**2).mean()

        optimizer_inr_latent.zero_grad()
        loss.backward()
        optimizer_inr_latent.step()

        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

# %%
T_train = dataset.t[-1]
t_test = torch.linspace(0.0, 1.5 * T_train, 200).to(device)
indices_interp = (t_test <= T_train).cpu()
indices_extrap = (t_test > T_train).cpu()

# Pick new unseen initial conditions
x0_tests = torch.tensor(np.random.uniform(-2.5, 2.5, size=(2, 2)), dtype=torch.float32).to(device)

plt.figure(figsize=(12, 6))

for i, x0 in enumerate(x0_tests):
    # GT
    gt = generate_trajectory(x0.cpu().numpy(), t_test.cpu().numpy(), noise_std=0.)
    t_train_tensor = torch.tensor(dataset.t, dtype=torch.float32).to(device)
    gt_train_part = torch.tensor(generate_trajectory(x0.cpu().numpy(), dataset.t), dtype=torch.float32).to(device)
    y_samples = gt_train_part
    
    # Model 1: Neural ODE (Generalizes by vector field)
    p_node = rollout(neural_ode, x0, t_test)
    
    # Model 2: Conditioned INR (Train-time code optimization)
    # optimize a latent for the trained-inr_latent model and evaluate
    z_opt_vis = optimize_latent(inr_latent, t_train_tensor, gt_train_part, latent_dim=latent_dim)
    p_inr_cond = rollout(inr_latent, z_opt_vis, t_test)[0]

    plt.subplot(1, 2, i + 1)
    plt.plot(gt[indices_interp, 0], gt[indices_interp, 1], "k-", label="GT")
    plt.plot(gt[indices_extrap, 0], gt[indices_extrap, 1], "k--", alpha=0.5, label="GT (Extrap)")
    plt.scatter(y_samples[:, 0], y_samples[:, 1], c='gray', alpha=0.5, s=15, label="Noisy Samples")
    plt.plot(p_node[:, 0], p_node[:, 1], color="blue", label="Neural ODE")
    plt.plot(p_inr_cond[:, 0], p_inr_cond[:, 1], color="orange", label="INR (Train-time optimized z)")
    
    plt.scatter(x0[0].cpu(), x0[1].cpu(), c="red", s=30)
    plt.axis("equal")
    plt.title(f"Test IC {i}")
    if i == 0: plt.legend(fontsize=7)

plt.tight_layout()
plt.show()
