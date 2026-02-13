# %% [markdown]
# # Structured State Space Models (S4-style)
#
# In this lab, you will explore continuous-time structured state space models (SSMs):
# - Implement a continuous-time diagonal SSM and discretize it
# - Train it in RNN-style by unrolling through time
# - Reinterpret the same model as a global convolution
# - Accelerate the convolution using FFT
# - Compare runtime performance between approaches

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
        self.dt = 10. / seq_len

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
            self.dt
        )

# Create a dataset and dataloader
dataset = ToyForecastDataset(n_samples=100, seq_len=1024)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


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
class DiagonalSSM(nn.Module):
    def __init__(self, d_in=1, d_hidden=32):
        super().__init__()
        self.d_hidden = d_hidden
        # Init. A, B, and C
        self.A = nn.Parameter(torch.randn(d_hidden))
        self.B = nn.Parameter(torch.randn(d_hidden, d_in) * 0.1)
        self.C = nn.Parameter(torch.randn(d_in, d_hidden) * 0.1)

    def A_d(self, dt):
        # TODO: Compute discrete A_d
        raise NotImplementedError()

    def B_d(self, dt):
        # TODO: Compute discrete B_d
        raise NotImplementedError()

    def forward(self, x, dt=0.1):
        """
        x: (B, L, d_model_in)
        dt: integration step size
        return: (B, L, d_model_out)
        """
        return self._forward_rnn(x, dt)

    def _forward_rnn(self, x, dt):
        """RNN-style forward: h[t+1] = A_d * h[t] + B_d * x[t]"""
        B, L, _ = x.shape
        A_d, B_d = self.A_d(dt), self.B_d(dt)
        h = torch.zeros(B, self.d_hidden, device=x.device)
        os = []
        for t in range(L):
            x_t = x[:, t]
            h = A_d * h + torch.matmul(x_t, B_d.transpose(0, 1))
            o = torch.matmul(h, self.C.t())
            os.append(o)
        return torch.stack(os, dim=1)

# %% + tags=["solution"]
class DiagonalSSM(nn.Module):
    def __init__(self, d_in=1, d_hidden=32):
        super().__init__()
        self.d_hidden = d_hidden
        # Init. A, B, and C
        self.A = nn.Parameter(torch.randn(d_hidden))
        self.B = nn.Parameter(torch.randn(d_hidden, d_in) * 0.1)
        self.C = nn.Parameter(torch.randn(d_in, d_hidden) * 0.1)

    def A_d(self, dt):
        return torch.exp(dt * self.A)

    def B_d(self, dt):
        A_d_val = self.A_d(dt)
        B_factor = (A_d_val - 1.0) / (self.A + 1e-5)
        return B_factor.unsqueeze(-1) * self.B

    def forward(self, x, dt=0.1):
        """
        x: (B, L, d_model_in)
        dt: integration step size
        return: (B, L, d_model_out)
        """
        return self._forward_rnn(x, dt)

    def _forward_rnn(self, x, dt):
        """RNN-style forward: h[t+1] = A_d * h[t] + B_d * x[t]"""
        B, L, _ = x.shape
        A_d, B_d = self.A_d(dt), self.B_d(dt)
        h = torch.zeros(B, self.d_hidden, device=x.device)
        os = []
        for t in range(L):
            x_t = x[:, t]
            h = A_d * h + torch.matmul(x_t, B_d.transpose(0, 1))
            o = torch.matmul(h, self.C.t())
            os.append(o)
        return torch.stack(os, dim=1)
    
# %% [markdown]
# **Question 3.** Now train your model on the following dataset.
# What do you observe?

# %%
def train(model, loader, epochs=10, lr=1e-3, forward_options=None):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for x, o, dt in loader:
            x, o, dt = x.to(device), o.to(device), dt.to(device)
            dt = dt[0].to(x.dtype)

            if forward_options is None:
                pred = model(x, dt)
            else:
                pred = model(x, dt, **forward_options)
            loss = ((pred - o) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(loader):.4f}")

# Initialize and train the RNN model
model = DiagonalSSM(d_in=1, d_hidden=32)
train(model, loader)

# %% [markdown]
# **Question 4.** What do you think is wrong with the current
# parametrization of your model? Fix that and check that it
# solves the explosion of h[t] along time.

# %%
class DiagonalSSM(nn.Module):
    """
    Continuous-time SSM:
        h'(t) = A h(t) + B x(t)
        o(t)  = C h(t)

    with diagonal A.

    Trained in RNN-style after discretization.
    """

    def __init__(self, d_in=1, d_hidden=32):
        super().__init__()

        self.d_hidden = d_hidden
        # Stable init
        # TODO here: define alpha
        # self.alpha = ...
        self.B = nn.Parameter(torch.randn(d_hidden, d_in) * 0.1)
        self.C = nn.Parameter(torch.randn(d_in, d_hidden) * 0.1)

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
        x: (B, L, d_model_in)
        dt: integration step size
        return: (B, L, d_model_out)
        """
        return self._forward_rnn(x, dt)

    def _forward_rnn(self, x, dt):
        """RNN-style forward: h[t+1] = A_d * h[t] + B_d * x[t]"""
        B, L, _ = x.shape
        A_d, B_d = self.A_d(dt), self.B_d(dt)
        h = torch.zeros(B, self.d_hidden, device=x.device)
        os = []
        for t in range(L):
            x_t = x[:, t]
            h = A_d * h + torch.matmul(x_t, B_d.transpose(0, 1))
            o = torch.matmul(h, self.C.t())
            os.append(o)
        return torch.stack(os, dim=1)

# %% + tags=["solution"]
class DiagonalSSM(nn.Module):
    """
    Diagonal SSM with stable A initialization.
    
    The model stores stable parameters alpha, B, and C.
    Implements RNN-style forward pass (unrolled discretized ODE).
    """
    def __init__(self, d_in=1, d_hidden=32):
        super().__init__()
        self.d_hidden = d_hidden
        # Stable init: A = -exp(alpha), where alpha is learnable
        self.alpha = nn.Parameter(torch.randn(d_hidden))
        self.B = nn.Parameter(torch.randn(d_hidden, d_in) * 0.1)
        self.C = nn.Parameter(torch.randn(d_in, d_hidden) * 0.1)

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
        """
        x: (B, L, d_model_in)
        dt: integration step size
        return: (B, L, d_model_out)
        """
        return self._forward_rnn(x, dt)

    def _forward_rnn(self, x, dt):
        """RNN-style forward: h[t+1] = A_d * h[t] + B_d * x[t]"""
        B, L, _ = x.shape
        A_d, B_d = self.A_d(dt), self.B_d(dt)
        h = torch.zeros(B, self.d_hidden, device=x.device)
        os = []
        for t in range(L):
            x_t = x[:, t]
            h = A_d * h + torch.matmul(x_t, B_d.transpose(0, 1))
            o = torch.matmul(h, self.C.t())
            os.append(o)
        return torch.stack(os, dim=1)

# Initialize and train the RNN model
model = DiagonalSSM(d_in=1, d_hidden=32)
train(model, loader)

# %% [markdown]
# ## Part 3: Convolution View
#
# **Question 5.** The same SSM can be viewed as a global convolution with kernel:
# ```
#     K_k = C A_d^k B_d
# ```
#
# Extend the `DiagonalSSM` class to support a conv-style forward pass
# that uses the kernel directly instead of unrolling through time.
# The code below is a baseline implementation you need to slightly extend (add the "conv" mode to the forward).
# Why can the powers of `A` be computed as `A_pows = A_d.unsqueeze(0) ** k_idx` (element-wise elevation to the power)?

# %%
class DiagonalSSM(nn.Module):
    """
    Diagonal SSM supporting both RNN-style and Conv-style forward passes.
    
    The model stores stable parameters alpha, B, and C. The forward method
    can be called with mode='rnn' (default) or mode='conv'.
    """
    def __init__(self, d_in=1, d_hidden=32):
        super().__init__()
        self.d_hidden = d_hidden
        # Stable init: A = -exp(alpha), where alpha is learnable
        self.alpha = nn.Parameter(torch.randn(d_hidden))
        self.B = nn.Parameter(torch.randn(d_hidden, d_in) * 0.1)
        self.C = nn.Parameter(torch.randn(d_in, d_hidden) * 0.1)

    @property
    def A(self):
        return -torch.exp(self.alpha)

    def A_d(self, dt):
        return torch.exp(dt * self.A)

    def B_d(self, dt):
        A_d_val = self.A_d(dt)
        B_factor = (A_d_val - 1.0) / (self.A + 1e-5)
        return B_factor.unsqueeze(-1) * self.B

    def forward(self, x, dt=0.1, mode='rnn'):
        """
        x: (B, L, d_model_in)
        dt: integration step size
        mode: 'rnn' or 'conv'
        return: (B, L, d_model_out)
        """
        # TODO: add the "conv" mode here
        if mode == 'rnn':
            return self._forward_rnn(x, dt)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'rnn' or 'conv'.")

    def _forward_rnn(self, x, dt):
        """RNN-style forward: h[t+1] = A_d * h[t] + B_d * x[t]"""
        B, L, _ = x.shape
        A_d, B_d = self.A_d(dt), self.B_d(dt)
        h = torch.zeros(B, self.d_hidden, device=x.device)
        os = []
        for t in range(L):
            x_t = x[:, t]
            h = A_d * h + torch.matmul(x_t, B_d.transpose(0, 1))
            o = torch.matmul(h, self.C.t())
            os.append(o)
        return torch.stack(os, dim=1)

    def _compute_kernel(self, L, dt):
        """Compute convolution kernel K[k] = C @ (A_d^k) @ B_d using vectorized operations."""
        A_d = self.A_d(dt)
        B_d = self.B_d(dt)
        C = self.C
        
        # Vectorized kernel computation using diagonal representation
        device = A_d.device
        dtype = A_d.dtype
        k_idx = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)  # (L, 1)
        A_pows = A_d.unsqueeze(0) ** k_idx  # (L, d_hidden)
        
        # terms: (L, d_hidden, d_model_in) = A_pows[..., None] * B_d[None, ...]
        terms = A_pows.unsqueeze(-1) * B_d.unsqueeze(0)
        
        # Ks[l] = C @ terms[l] -> (L, d_model_out, d_model_in)
        Ks = torch.einsum("os, lsi -> loi", C, terms)
        return Ks

    def _forward_conv(self, x, dt):
        """
        Conv-style forward using torch.nn.functional.conv1d.
        x: (B, L, d_in)
        """
        B, L, d_in = x.shape
        
        # 1. Compute the kernel: (L, d_model_out, d_model_in)
        K = self._compute_kernel(L, dt) 
        
        # 2. Prepare x for conv1d: (B, d_model_in, L)
        x_t = x.transpose(1, 2)
        
        # 3. Prepare K for conv1d: (d_model_out, d_model_in, kernel_size)
        K_t = K.permute(1, 2, 0)
        
        # 4. Causal Padding: Add (kernel_size - 1) to the LEFT side of the temporal dim
        x_padded = torch.nn.functional.pad(x_t, (L - 1, 0))
        
        # 5. Apply Convolution
        # We flip the kernel when using conv1d because it performs cross-correlation.
        out = torch.nn.functional.conv1d(x_padded, torch.flip(K_t, dims=[-1]))
        
        # 6. Back to (B, L, d_model_out)
        return out.transpose(1, 2)

# %% + tags=["solution"]
class DiagonalSSM(nn.Module):
    """
    Diagonal SSM supporting both RNN-style and Conv-style forward passes.
    
    The model stores stable parameters alpha, B, and C. The forward method
    can be called with mode='rnn' (default) or mode='conv'.
    """
    def __init__(self, d_in=1, d_hidden=32):
        super().__init__()
        self.d_hidden = d_hidden
        # Stable init: A = -exp(alpha), where alpha is learnable
        self.alpha = nn.Parameter(torch.randn(d_hidden))
        self.B = nn.Parameter(torch.randn(d_hidden, d_in) * 0.1)
        self.C = nn.Parameter(torch.randn(d_in, d_hidden) * 0.1)

    @property
    def A(self):
        return -torch.exp(self.alpha)

    def A_d(self, dt):
        return torch.exp(dt * self.A)

    def B_d(self, dt):
        A_d_val = self.A_d(dt)
        B_factor = (A_d_val - 1.0) / (self.A + 1e-5)
        return B_factor.unsqueeze(-1) * self.B

    def forward(self, x, dt=0.1, mode='rnn'):
        """
        x: (B, L, d_model_in)
        dt: integration step size
        mode: 'rnn' or 'conv'
        return: (B, L, d_model_out)
        """
        if mode == 'rnn':
            return self._forward_rnn(x, dt)
        elif mode == 'conv':
            return self._forward_conv(x, dt)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'rnn' or 'conv'.")

    def _forward_rnn(self, x, dt):
        """RNN-style forward: h[t+1] = A_d * h[t] + B_d * x[t]"""
        B, L, _ = x.shape
        A_d, B_d = self.A_d(dt), self.B_d(dt)
        h = torch.zeros(B, self.d_hidden, device=x.device)
        os = []
        for t in range(L):
            x_t = x[:, t]
            h = A_d * h + torch.matmul(x_t, B_d.transpose(0, 1))
            o = torch.matmul(h, self.C.t())
            os.append(o)
        return torch.stack(os, dim=1)

    def _compute_kernel(self, L, dt):
        """Compute convolution kernel K[k] = C @ (A_d^k) @ B_d using vectorized operations."""
        A_d = self.A_d(dt)
        B_d = self.B_d(dt)
        C = self.C
        
        # Vectorized kernel computation using diagonal representation
        device = A_d.device
        dtype = A_d.dtype
        k_idx = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)  # (L, 1)
        A_pows = A_d.unsqueeze(0) ** k_idx  # (L, d_hidden)
        
        # terms: (L, d_hidden, d_model_in) = A_pows[..., None] * B_d[None, ...]
        terms = A_pows.unsqueeze(-1) * B_d.unsqueeze(0)
        
        # Ks[l] = C @ terms[l] -> (L, d_model_out, d_model_in)
        Ks = torch.einsum("os, lsi -> loi", C, terms)
        return Ks

    def _forward_conv(self, x, dt):
        """
        Conv-style forward using torch.nn.functional.conv1d.
        x: (B, L, d_in)
        """
        B, L, d_in = x.shape
        
        # 1. Compute the kernel: (L, d_model_out, d_model_in)
        K = self._compute_kernel(L, dt) 
        
        # 2. Prepare x for conv1d: (B, d_model_in, L)
        x_t = x.transpose(1, 2)
        
        # 3. Prepare K for conv1d: (d_model_out, d_model_in, kernel_size)
        K_t = K.permute(1, 2, 0)
        
        # 4. Causal Padding: Add (kernel_size - 1) to the LEFT side of the temporal dim
        x_padded = torch.nn.functional.pad(x_t, (L - 1, 0))
        
        # 5. Apply Convolution
        # We flip the kernel when using conv1d because it performs cross-correlation.
        out = torch.nn.functional.conv1d(x_padded, torch.flip(K_t, dims=[-1]))
        
        # 6. Back to (B, L, d_model_out)
        return out.transpose(1, 2)

# %% [markdown]
# ## Part 4: FFT Convolution
#
# Convolution in time domain becomes element-wise multiplication in
# frequency domain. Below is the FFT-based implementation of convolution using:
# ```
#     O = IFFT(FFT(X) ⊙ FFT(K))
# ```
#
# where ⊙ denotes element-wise multiplication.
# **Question 6.** Using this implementation, add an "fft-conv" mode to the forward 
# of your model that would rely on this trick rather than computing the actual convolution.

# %%
def fft_convolve(x, K):
    """
    x: (B, L, d_in)
    K: (L, d_in, d_out)
    """

    B, L, d = x.shape
    fft_size = 2 * L

    x_f = torch.fft.rfft(x, n=fft_size, dim=1)
    K_f = torch.fft.rfft(K, n=fft_size, dim=0)

    o_f = torch.einsum("bli, lio -> blo", x_f, K_f)
    o = torch.fft.irfft(o_f, n=fft_size, dim=1)

    return o[:, :L]

# %% + tags=["solution"]
class DiagonalSSM(nn.Module):
    """
    Diagonal SSM supporting RNN-style, Conv-style, and FFT-based convolution.
    
    The forward method can be called with mode='rnn' (default), mode='conv', 
    or mode='fft-conv' for FFT-accelerated convolution.
    """
    def __init__(self, d_in=1, d_hidden=32):
        super().__init__()
        self.d_hidden = d_hidden
        # Stable init: A = -exp(alpha), where alpha is learnable
        self.alpha = nn.Parameter(torch.randn(d_hidden))
        self.B = nn.Parameter(torch.randn(d_hidden, d_in) * 0.1)
        self.C = nn.Parameter(torch.randn(d_in, d_hidden) * 0.1)

    @property
    def A(self):
        return -torch.exp(self.alpha)

    def A_d(self, dt):
        return torch.exp(dt * self.A)

    def B_d(self, dt):
        A_d_val = self.A_d(dt)
        B_factor = (A_d_val - 1.0) / (self.A + 1e-5)
        return B_factor.unsqueeze(-1) * self.B

    def forward(self, x, dt=0.1, mode='rnn'):
        """
        x: (B, L, d_in)
        dt: integration step size
        mode: 'rnn', 'conv', or 'fft-conv'
        return: (B, L, d_out)
        """
        if mode == 'rnn':
            return self._forward_rnn(x, dt)
        elif mode == 'conv':
            return self._forward_conv(x, dt)
        elif mode == 'fft-conv':
            return self._forward_fft_conv(x, dt)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'rnn', 'conv', or 'fft-conv'.")

    def _forward_rnn(self, x, dt):
        """RNN-style forward: h[t+1] = A_d * h[t] + B_d * x[t]"""
        B, L, _ = x.shape
        A_d, B_d = self.A_d(dt), self.B_d(dt)
        h = torch.zeros(B, self.d_hidden, device=x.device)
        os = []
        for t in range(L):
            x_t = x[:, t]
            h = A_d * h + torch.matmul(x_t, B_d.transpose(0, 1))
            o = torch.matmul(h, self.C.t())
            os.append(o)
        return torch.stack(os, dim=1)

    def _compute_kernel(self, L, dt):
        """Compute convolution kernel K[k] = C @ (A_d^k) @ B_d using vectorized operations."""
        A_d = self.A_d(dt)
        B_d = self.B_d(dt)
        C = self.C
        
        # Vectorized kernel computation using diagonal representation
        device = A_d.device
        dtype = A_d.dtype
        k_idx = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)  # (L, 1)
        A_pows = A_d.unsqueeze(0) ** k_idx  # (L, d_hidden)
        
        # terms: (L, d_hidden, d_model_in) = A_pows[..., None] * B_d[None, ...]
        terms = A_pows.unsqueeze(-1) * B_d.unsqueeze(0)
        
        # Ks[l] = C @ terms[l] -> (L, d_model_in, d_model_out)
        Ks = torch.einsum("os, lsi -> lio", C, terms)
        return Ks

    def _forward_conv(self, x, dt):
        """
        Conv-style forward using torch.nn.functional.conv1d.
        x: (B, L, d_model_in)
        """
        B, L, d_model_in = x.shape
        
        # 1. Compute the kernel: (L, d_model_in, d_model_out)
        K = self._compute_kernel(L, dt) 
        
        # 2. Prepare x for conv1d: (B, d_model_in, L)
        x_t = x.transpose(1, 2)
        
        # 3. Causal Padding: Add (L - 1) to the LEFT side of the temporal dim
        x_padded = torch.nn.functional.pad(x_t, (L - 1, 0))
        
        # 4. Apply Convolution
        # We flip the kernel when using conv1d because it performs cross-correlation.
        out = torch.nn.functional.conv1d(x_padded, torch.flip(K, dims=[-1]))
        
        # 6. Back to (B, L, d_model_out)
        return out.transpose(1, 2)

    def _forward_fft_conv(self, x, dt):
        """FFT-based convolution for faster computation."""
        B, L, d = x.shape
        K = self._compute_kernel(L, dt)  # (L, d_model_in, d_model_out)
        return fft_convolve(x, K)

# %% [markdown]
# ## Part 5: Training and Evaluation
#
# **Question 7.** Train the RNN-style and Conv-style versions on the toy dataset.
# Is there a clear winner in terms of final loss?

# %% + tags=["solution"]
modes = ["rnn", "conv", "fft-conv"]

for mode in modes:
    print(f"\nTraining SSM with mode {mode}")
    model = DiagonalSSM()
    train(model, loader, forward_options={"mode": mode})

# %% [markdown]
# **Question 8.** Compare the runtime performance of RNN-style vs Conv-style SSM at inference.
# Which one is faster? Can you explain why?

# %%
x, _, dt = next(iter(loader))
x = x.to(device)
dt = dt.to(device)[0].to(x.dtype)
n_repeats = 1000

print("\nTiming comparison (forward pass)...")

# TODO here

print("\nDone.")

# %% + tags=["solution"]
x, _, dt = next(iter(loader))
x = x.to(device)
dt = dt.to(device)[0].to(x.dtype)
n_repeats = 1000

print("\nTiming comparison (forward pass)...")

modes = ["rnn", "conv", "fft-conv"]

for mode in modes:
    start = time.time()
    for _ in range(n_repeats):
        _ = model(x, dt, mode=mode)
    print(f"Timings (mode {mode}):", (time.time() - start) / n_repeats)

print("\nDone.")
