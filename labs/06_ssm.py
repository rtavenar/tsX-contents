# %% [markdown]
# # Structured State Space Models (S4-style)
#
# In this lab, you will explore continuous-time structured state space models (SSMs):
# - Implement a continuous-time diagonal SSM and discretize it
# - Train it in RNN-style by unrolling through time
# - Reinterpret the same model as a global convolution
# - Accelerate the convolution using FFT
# - Compare runtime performance between approaches

# TODO-Romain: the current content is sufficient, now the goal is to make the lab as accessible as possible to the students and copy-paste some content from one place to the other

# %%
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## Part 1: Toy Forecasting Dataset
#
# **Question 1.** Understand the `ToyForecastDataset` class which generates
# simple sinusoidal forecasting data. We will use this to train and evaluate
# our different SSM implementations.

# %%
class ToyForecastDataset(Dataset):
    """
    Simple sinusoidal forecasting dataset.
    Predict next timestep.
    """
    def __init__(self, n_samples=1000, seq_len=128):
        self.seq_len = seq_len
        self.data = []

        for _ in range(n_samples):
            t = np.linspace(0, 10, seq_len + 1)
            signal = (
                np.sin(2 * t)
                + 0.5 * np.sin(5 * t)
                + 0.1 * np.random.randn(len(t))
            )
            self.data.append(signal.astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return (
            torch.tensor(data[:-1]).unsqueeze(-1),
            torch.tensor(data[1:]).unsqueeze(-1),
        )

# %% [markdown]
# ## Part 2: Continuous-time SSM (Diagonal A)
#
# **Question 2.** Implement the discretization method for a continuous-time diagonal SSM.
# The continuous-time system is:
# ```
#     h'(t) = A h(t) + B x(t)
#     o(t)  = C h(t)
# ```
#
# with diagonal A. We discretize using:
# ```
#     A_d = exp(dt A)
#     B_d = (exp(dt A) - 1) / A * B
# ```
#
# Then train it in RNN-style by unrolling:
# ```
#     h[t+1] = A_d * h[t] + B_d * x[t]
#     o[t]   = C * h[t]
# ```

# %%
class DiagonalSSM_RNN(nn.Module):
    def __init__(self, d_model=1, d_state=32):
        super().__init__()
        self.d_state = d_state
        # Init. A, B, and C
        self.A = nn.Parameter(torch.randn(d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

    def A_d(self, dt):
        # TODO: Compute discrete A_d
        raise NotImplementedError()

    def B_d(self, dt):
        # TODO: Compute discrete B_d
        raise NotImplementedError()

    def forward(self, x, dt=0.1):
        # TODO: RNN-style forward
        raise NotImplementedError()
    
# %% [markdown]
# **Question 3.** Now train your model on the following dataset.
# What do you observe?

# %%
# TODO here: code for training a model on the dataset

# %% [markdown]
# **Question 4.** What do you think is wrong with the current
# parametrization of your model? Fix that and check that it
# solves the explosion of h[t] along time.

# %%
class DiagonalSSM_RNN(nn.Module):
    """
    Continuous-time SSM:
        h'(t) = A h(t) + B x(t)
        o(t)  = C h(t)

    with diagonal A.

    Trained in RNN-style after discretization.
    """

    def __init__(self, d_model=1, d_state=32):
        super().__init__()

        self.d_state = d_state
        # Stable init
        # TODO here: define alpha
        # self.alpha = ...
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

    @property
    def A(self):
        # TODO here: compute A based on alpha such that A is 
        # constrained to be a vector of negative values
        raise NotImplementedError("Implement A property.")

    def A_d(self, dt):
        """
        Compute discrete A_d.
        A_d = exp(dt A)
        
        TODO: Implement discretization.
        """
        raise NotImplementedError("Implement A_d method.")

    def B_d(self, dt):
        """
        Compute discrete B_d.
        B_d = (exp(dt A) - 1)/A * B
        
        TODO: Implement discretization.
        """
        raise NotImplementedError("Implement B_d method.")

    def forward(self, x, dt=0.1):
        """
        x: (B, L, d_model)
        dt: integration step size
        return: (B, L, d_model)
        
        TODO: Implement RNN-style forward pass using self.A_d(dt) and self.B_d(dt).
        """
        raise NotImplementedError("Implement RNN forward pass.")

# %% + tags=["solution"]
class DiagonalSSM_RNN(nn.Module):
    def __init__(self, d_model=1, d_state=32):
        super().__init__()
        self.d_state = d_state
        # Stable init
        self.alpha = nn.Parameter(torch.randn(d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

    @property
    def A(self):
        return -torch.exp(self.alpha)

    def A_d(self, dt):
        return torch.exp(dt * self.A)

    def B_d(self, dt):
        A_d_val = self.A_d(dt)
        B_factor = (A_d_val - 1.0) / (self.A + 1e-5)
        return B_factor.unsqueeze(-1) * self.B

    def forward(self, x, dt=0.1):
        B, L, _ = x.shape
        A_d, B_d = self.A_d(dt), self.B_d(dt)
        h = torch.zeros(B, self.d_state, device=x.device)
        os = []
        for t in range(L):
            x_t = x[:, t]
            h = A_d * h + torch.matmul(x_t, B_d.transpose(0, 1))
            o = torch.matmul(h, self.C.t())
            os.append(o)
        return torch.stack(os, dim=1)

# %% [markdown]
# ## Part 3: Convolution View
#
# **Question 5.** The same SSM can be viewed as a global convolution with kernel:
# ```
#     K_k = C A_d^k B_d
# ```
#
# Implement the `compute_kernel` method to produce the convolution kernel,
# then use it in the `forward` method via `nn.functional.conv1d`.

# %%
class DiagonalSSM_Conv(nn.Module):
    """
    Uses convolution kernel:

        K_k = C A_d^k B_d
    """

    def __init__(self, d_model=1, d_state=32):
        super().__init__()
        self.ssm = DiagonalSSM_RNN(d_model, d_state)

    def compute_kernel(self, L, dt=0.1):
        """
        Return kernel of shape (L, d_model, d_model)
        
        TODO: Implement kernel computation using self.ssm.A_d(dt) and self.ssm.B_d(dt).
        """
        A_d = self.ssm.A_d(dt)
        B_d = self.ssm.B_d(dt)
        C = self.ssm.C

        Ks = []
        A_power = torch.ones_like(A_d)

        for _ in range(L):
            term = C @ (A_power.unsqueeze(-1) * B_d)
            Ks.append(term)
            A_power = A_power * A_d

        return torch.stack(Ks, dim=0)

    def forward(self, x, dt=0.1):
        B, L, d = x.shape
        K = self.compute_kernel(L, dt)  # (L, d_model, d_model)

        # Causal convolution: o[t] = sum_{k=0}^{t} K[k] @ x[t-k]
        # Vectorized using torch operations
        
        # Pad x on the left to create a window view
        x_pad = nn.functional.pad(x, (0, 0, L - 1, 0))  # (B, 2L-1, d)
        
        # Create causal window indices: for each t, get x[t:t+L] reversed
        # indices[t, k] should point to x_pad[t + (L-1) - k]
        t_idx = torch.arange(L, device=x.device).unsqueeze(1)  # (L, 1)
        k_idx = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        indices = (L - 1) + t_idx - k_idx  # (L, L)
        indices = indices.clamp(0, 2 * L - 2)
        
        # Gather windows: (B, 2L-1, d) -> (B, L, L, d)
        x_windows = x_pad[:, indices, :]  # (B, L, L, d)
        
        # Apply kernel via einsum
        # x_windows: (B, t_pos, k_idx, d_in)
        # K: (L, d_out, d_in)
        # Result: (B, t_pos, d_out)
        K_t = K.transpose(1, 2)  # (L, d_in, d_out)
        out = torch.einsum('btld, ldo -> bto', x_windows, K_t)
        
        return out

# %% + tags=["solution"]
class DiagonalSSM_Conv_Vectorized(nn.Module):
    """
    Uses convolution kernel:

        K_k = C A_d^k B_d
    """

    def __init__(self, d_model=1, d_state=32):
        super().__init__()
        self.ssm = DiagonalSSM_RNN(d_model, d_state)

    def compute_kernel(self, L, dt=0.1):
        """
        Return kernel of shape (L, d_model, d_model)
        """

        A_d = self.ssm.A_d(dt)
        B_d = self.ssm.B_d(dt)
        C = self.ssm.C
        # Vectorized kernel computation using diagonal representation.
        # A_d: (d_state,), B_d: (d_state, d_model_in), C: (d_model_out, d_state)
        # Build A_pows: (L, d_state) where A_pows[k, s] = A_d[s] ** k
        device = A_d.device
        dtype = A_d.dtype
        k_idx = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)  # (L,1)
        A_pows = A_d.unsqueeze(0) ** k_idx  # (L, d_state)

        # terms: (L, d_state, d_model_in) = A_pows[...,None] * B_d[None,...]
        terms = A_pows.unsqueeze(-1) * B_d.unsqueeze(0)

        # Ks[l]_{o,in} = sum_s C[o,s] * terms[l,s,in]
        Ks = torch.einsum("os, lsi -> loi", C, terms)
        return Ks

    def forward(self, x, dt=0.1):
        B, L, d = x.shape
        K = self.compute_kernel(L, dt)  # (L, d_model, d_model)

        # Causal convolution: o[t] = sum_{k=0}^{t} K[k] @ x[t-k]
        # Vectorized using torch operations (no Python loops)
        
        # Pad x on the left to create a window view
        x_pad = nn.functional.pad(x, (0, 0, L - 1, 0))  # (B, 2L-1, d)
        
        # Create causal window indices: for each t, get x[t:t+L] reversed
        # indices[t, k] should point to x_pad[t + (L-1) - k]
        t_idx = torch.arange(L, device=x.device).unsqueeze(1)  # (L, 1)
        k_idx = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        indices = (L - 1) + t_idx - k_idx  # (L, L)
        indices = indices.clamp(0, 2 * L - 2)
        
        # Gather windows: (B, 2L-1, d) -> (B, L, L, d)
        x_windows = x_pad[:, indices, :]  # (B, L, L, d)
        
        # Apply kernel via einsum
        # x_windows: (B, t_pos, k_idx, d_in)
        # K: (L, d_out, d_in)
        # Result: (B, t_pos, d_out)
        K_t = K.transpose(1, 2)  # (L, d_in, d_out)
        out = torch.einsum('btld, ldo -> bto', x_windows, K_t)
        
        return out

# %% [markdown]
# ## Part 4: FFT Convolution
#
# **Question 6.** Convolution in time domain becomes element-wise multiplication in
# frequency domain. Implement FFT-based convolution to accelerate computations:
# ```
#     O = IFFT(FFT(X) ⊙ FFT(K))
# ```
#
# where ⊙ denotes element-wise multiplication.

# %%
def fft_convolve(x, K):
    """
    x: (B, L, d)
    K: (L, d, d)
    
    TODO: Implement FFT convolution.
    """
    raise NotImplementedError("Implement FFT convolution.")

# %% + tags=["solution"]
def fft_convolve(x, K):
    """
    x: (B, L, d)
    K: (L, d, d)
    """

    B, L, d = x.shape
    fft_size = 2 * L

    x_f = torch.fft.rfft(x, n=fft_size, dim=1)
    K_f = torch.fft.rfft(K, n=fft_size, dim=0)

    o_f = torch.einsum("bld,ldd->bld", x_f, K_f)
    o = torch.fft.irfft(o_f, n=fft_size, dim=1)

    return o[:, :L]

# %% [markdown]
# ## Part 5: Training and Evaluation
#
# **Question 7.** Implement the `train` function to train your SSM models.
# Then train both the RNN-style and Conv-style versions on the toy dataset.
# Compare their training curves and runtime performance.

# %%
def train(model, loader, epochs=10, lr=1e-3, dt=0.1):
    """
    TODO: Implement training loop using model(x, dt).
    """
    raise NotImplementedError("Implement training loop.")

# %% + tags=["solution"]
def train(model, loader, epochs=10, lr=1e-3, dt=0.1):

    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for x, o in loader:
            x, o = x.to(device), o.to(device)

            pred = model(x, dt)
            loss = ((pred - o) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(loader):.4f}")

# %%
dataset = ToyForecastDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("\nTraining RNN-style SSM")
model_rnn = DiagonalSSM_RNN()
train(model_rnn, loader)
print("\nTraining Conv-style SSM (non-optimized)")
model_conv_unopt = DiagonalSSM_Conv()
train(model_conv_unopt, loader)

print("\nTraining Conv-style SSM (vectorized)")
model_conv_opt = DiagonalSSM_Conv_Vectorized()
train(model_conv_opt, loader)

# %% [markdown]
# **Question 8.** Compare the runtime performance of RNN-style vs Conv-style SSM.
# Which one is faster? Can you explain why?

# %%
x, _ = next(iter(loader))
x = x.to(device)
n_repeats = 1000
dt = 0.1

print("\nTiming comparison (forward pass)...")
 
start = time.time()
for _ in range(n_repeats):
    _ = model_rnn(x, dt)
print("RNN time:", (time.time() - start) / n_repeats)

start = time.time()
for _ in range(n_repeats):
    _ = model_conv_unopt(x, dt)
print("Conv (non-optimized) time:", (time.time() - start) / n_repeats)

start = time.time()
for _ in range(n_repeats):
    _ = model_conv_opt(x, dt)
print("Conv (vectorized) time:", (time.time() - start) / n_repeats)

print("\nDone.")
