# %% [markdown]
# # Time Series Classification
# 
# In this lab, you will work on time series classification (TSC) tasks. You will learn how to:
# - Load the UCI HAR dataset and build PyTorch data loaders
# - Build a simple data loader for classification tasks
# - Implement TimesNet: a modern deep learning model based on 2D convolutions
#
# The lab focuses on practical implementation, allowing you to understand the key components
# of TSC models without getting lost in implementation details.

# %%
import numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Part 1: Data Loading
# 
# In this section, you will load the UCI HAR (Human Activity Recognition) dataset
# and create a PyTorch data loader for it.

# %% [markdown]
# **Question 1.** Download the UCI HAR dataset (if needed) and load the train/test arrays.
# 
# - Download: `wget -O data/UCIHAR.npz https://github.com/rtavenar/ml-datasets/releases/download/UCIHAR/UCIHAR.npz`
# - Load from `.npz` — data is `(n_samples, T, C)` (time steps, channels)

# %%
!wget -O data/UCIHAR.npz https://github.com/rtavenar/ml-datasets/releases/download/UCIHAR/UCIHAR.npz
# %% + tags=["solution"]
dataset = numpy.load("data/UCIHAR.npz")
X_train, y_train = dataset["X_train"], dataset["y_train"].ravel()
X_test, y_test = dataset["X_test"], dataset["y_test"].ravel()

print(f"Dataset: UCI HAR — X shape (n_samples, T, C): {X_train.shape}")
print(f"Train labels: {numpy.unique(y_train)}, Classes: {len(numpy.unique(y_train))}")

# %% [markdown]
# **Question 2.** Implement a PyTorch `Dataset` class for time series classification.
#
# The dataset should:
# - Take numpy arrays `X` and `y` as input
# - Assume `X` has shape `(n_samples, T, C)` — samples, time steps, channels
# - Return `(time_series, label)` with `time_series` of shape `(T, C)`
# - Optionally support normalization

# %% + tags=["solution"]
class TSCDataset(torch.utils.data.Dataset):
    """Time Series Classification Dataset. Expects X of shape (n_samples, T, C)."""
    
    def __init__(self, X, y, normalize=True, scaler=None):
        super().__init__()
        self.X = X.astype(numpy.float32)
        self.y = y.astype(numpy.int64)
        self.normalize = normalize
        if normalize:
            self.scaler = scaler or StandardScaler()
            if scaler is None:
                self.scaler.fit(self.X.reshape(-1, 1))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        ts = self.X[idx]  # (T, C)
        if self.normalize:
            ts = self.scaler.transform(ts.reshape(-1, 1)).reshape(ts.shape)
        return torch.from_numpy(ts), torch.tensor(self.y[idx], dtype=torch.long)

# Create datasets and data loaders (fit scaler on train only)
train_dataset = TSCDataset(X_train, y_train, normalize=True)
test_dataset = TSCDataset(X_test, y_test, normalize=True, scaler=train_dataset.scaler)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

# Verify shapes
sample_ts, sample_label = train_dataset[0]
print(f"Sample time series shape: {sample_ts.shape}")
print(f"Sample label: {sample_label.item()}")

# %% [markdown]
# ## Part 2: TimesNet Model
# 
# TimesNet transforms time series into 2D representations based on detected periods,
# then applies 2D convolutions. We provide two building blocks:
#
# 1. **`extract_periods_and_amplitudes`** — FFT-based period detection
# 2. **`PeriodReshape`** — nn.Module that reshapes `(B, T, C)` into 2D tensors given periods

# %%
def extract_periods_and_amplitudes(ts, top_k=5):
    """
    Extract top-k periods and their FFT amplitudes from a time series.
    
    Args:
        ts: tensor of shape (T,) or (T, C) — 1D or multivariate (channels averaged for FFT)
        top_k: number of periods to return
    
    Returns:
        periods: list of int, period lengths
        amplitudes: list of float, FFT power at each period
    """
    if ts.dim() > 1:
        ts = ts.mean(dim=-1)  # (T,)
    ts = ts.float()
    fft_vals = torch.fft.rfft(ts)
    power = torch.abs(fft_vals) ** 2
    
    min_period, max_period = 2, len(ts) // 2
    freqs = torch.arange(len(power), dtype=torch.float32, device=ts.device)
    periods_float = len(ts) / (freqs + 1e-8)
    
    valid = (periods_float >= min_period) & (periods_float <= max_period)
    pwr, prd = power[valid], periods_float[valid]
    if len(pwr) == 0:
        return [len(ts) // 2], [1.0]
    
    k = min(top_k, len(pwr))
    vals, idx = torch.topk(pwr, k)
    return prd[idx].int().tolist(), vals.tolist()


class PeriodReshape(nn.Module):
    """
    Reshape multivariate time series (B, T, C) into 2D tensors per period.
    Output: list of tensors of shape (B, C, period, num_periods).
    """

    def __init__(self, periods):
        """
        Args:
            periods: list or tensor of int, period lengths to use
        """
        super().__init__()
        self.periods = [int(p) for p in periods]

    def forward(self, x):
        """
        Args:
            x: (B, T, C) — batch of multivariate time series
        Returns:
            list of (B, C, period, num_periods) tensors, one per period
        """
        B, T, C = x.shape
        out = []
        for p in self.periods:
            n = (T + p - 1) // p
            need = n * p
            if need > T:
                xp = torch.nn.functional.pad(x, (0, 0, 0, need - T))
            else:
                xp = x[:, :need]
            # (B, T', C) -> (B, C, p, n)
            out.append(xp.permute(0, 2, 1).reshape(B, C, p, n))
        return out

# %% [markdown]
# **Question 3.** Use `extract_periods_and_amplitudes` on a sample to get periods, then
# use `PeriodReshape` to obtain 2D representations. Visualize one of them.

# %% + tags=["solution"]
sample_ts = train_dataset[0][0]  # (T, C)
periods, amplitudes = extract_periods_and_amplitudes(sample_ts, top_k=3)
print(f"Detected periods: {periods}, amplitudes: {amplitudes}")

reshape_module = PeriodReshape(periods)
# Add batch dim: (T, C) -> (1, T, C)
reps = reshape_module(sample_ts.unsqueeze(0))
for i, r in enumerate(reps):
    print(f"Period {periods[i]}: 2D shape {r.shape}")

plt.figure(figsize=(10, 4))
plt.imshow(reps[0][0].mean(0).numpy(), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title(f"2D Representation (Period {periods[0]})")
plt.xlabel("Period index")
plt.ylabel("Time within period")
plt.show()

# %% [markdown]
# **Question 4.** Implement a TimesNet model for classification.
#
# Use `PeriodReshape` and implement:
# - 2D convolutional blocks on each period’s representation
# - Global pooling, feature aggregation, and a classification head
#
# Hint: periods can be fixed (e.g. from `extract_periods_and_amplitudes` on training data).

# %%
class TimesNetBlock(nn.Module):
    """2D conv block on period-reshaped data. in_channels = C (input channels)."""

    def __init__(self, in_channels, out_channels=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


# %% + tags=["solution"]
class TimesNet(nn.Module):
    """TimesNet for classification. Uses fixed periods and PeriodReshape."""

    def __init__(self, n_channels, num_classes, periods, hidden_dim=64, num_blocks=2):
        super().__init__()
        self.reshape = PeriodReshape(periods)
        self.blocks = nn.ModuleList([
            nn.Sequential(*[
                TimesNetBlock(
                    in_channels=n_channels if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                )
                for i in range(num_blocks)
            ])
            for _ in periods
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(periods), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        reps = self.reshape(x)  # list of (B, C, p, n)
        feats = []
        for r, blk in zip(reps, self.blocks):
            h = blk(r)
            feats.append(self.pool(h).flatten(1))
        out = torch.cat(feats, dim=1)
        return self.classifier(out)


# Compute periods once from training data
_ref_ts = torch.from_numpy(X_train.mean(axis=0)).float()  # (T, C)
periods, _ = extract_periods_and_amplitudes(_ref_ts, top_k=3)
num_classes = len(numpy.unique(y_train))
n_channels = X_train.shape[2]

model = TimesNet(
    n_channels=n_channels,
    num_classes=num_classes,
    periods=periods,
    hidden_dim=64,
    num_blocks=2,
)

# Test forward pass
sample_batch, _ = next(iter(train_loader))
with torch.no_grad():
    logits = model(sample_batch)
    print(f"Input shape: {sample_batch.shape}")
    print(f"Output logits shape: {logits.shape}")

# %% [markdown]
# **Question 5.** Train and evaluate the TimesNet model using the loops provided below.

# %%
def train_epoch(model, dataloader, optimizer, criterion, device='cpu'):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for ts, labels in dataloader:
        ts, labels = ts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(ts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * ts.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device='cpu'):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for ts, labels in dataloader:
        ts, labels = ts.to(device), labels.to(device)
        
        logits = model(ts)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * ts.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def train_classif_model(model, train_loader, test_loader, n_epochs=50, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    return {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'test_loss': test_losses,
        'test_acc': test_accs
    }

# %% + tags=["solution"]

# Train model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

logs = train_classif_model(model, train_loader, test_loader, n_epochs=5, lr=1e-3, device=device)

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(logs['train_loss'], label='Train Loss')
plt.plot(logs['test_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')

plt.subplot(1, 2, 2)
plt.plot(logs['train_acc'], label='Train Accuracy')
plt.plot(logs['test_acc'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Test Accuracy')

plt.tight_layout()
plt.show()

# Final results
print(f"\nFinal Results:")
print(f"TimesNet - Test Accuracy: {logs['test_acc'][-1]:.4f}")

# %% [markdown]
# **Question 6.** In the TimesNet paper, the authors suggest using a **weighted average** for
# aggregating period features instead of concatenation: the weights are a softmax over the
# period amplitudes (higher FFT power → higher weight).
#
# Implement an alternative `TimesNetAmplitudeWeighted` model that:
# - Takes `periods` and `amplitudes` (from `extract_periods_and_amplitudes`) at init
# - Pools each period's features to `(B, hidden_dim)` as before
# - Aggregates them with a **softmax-weighted average**: weights = softmax(amplitudes)
# - Uses a classifier on the aggregated `(B, hidden_dim)` representation
#
# Train and compare its test accuracy with the standard TimesNet from Question 5.

# %% + tags=["solution"]
class TimesNetAmplitudeWeighted(nn.Module):
    """TimesNet with amplitude-weighted aggregation instead of concatenation."""

    def __init__(self, n_channels, num_classes, periods, amplitudes, hidden_dim=64, num_blocks=2):
        super().__init__()
        self.reshape = PeriodReshape(periods)
        self.blocks = nn.ModuleList([
            nn.Sequential(*[
                TimesNetBlock(
                    in_channels=n_channels if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                )
                for i in range(num_blocks)
            ])
            for _ in periods
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Softmax over amplitudes as fixed weights (buffer = non-trainable, moves with model)
        amps = torch.tensor(amplitudes, dtype=torch.float32)
        self.register_buffer("weights", torch.softmax(amps, dim=0))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        reps = self.reshape(x)
        feats = []
        for r, blk in zip(reps, self.blocks):
            h = blk(r)
            feats.append(self.pool(h).flatten(1))  # (B, hidden_dim)
        # Weighted average: (B, hidden_dim)
        stacked = torch.stack(feats, dim=1)  # (B, n_periods, hidden_dim)
        out = (stacked * self.weights.view(1, -1, 1)).sum(dim=1)
        return self.classifier(out)

# %%
# Get amplitudes from reference (same as for periods)
_ref_ts = torch.from_numpy(X_train.mean(axis=0)).float()
periods_aw, amplitudes_aw = extract_periods_and_amplitudes(_ref_ts, top_k=3)

model_aw = TimesNetAmplitudeWeighted(
    n_channels=n_channels,
    num_classes=num_classes,
    periods=periods_aw,
    amplitudes=amplitudes_aw,
    hidden_dim=64,
    num_blocks=2,
)

print("Training TimesNetAmplitudeWeighted...")
logs_aw = train_classif_model(model_aw, train_loader, test_loader, n_epochs=5, lr=1e-3, device=device)

print(f"\nComparison:")
print(f"TimesNet (concat):        Test Acc = {logs['test_acc'][-1]:.4f}")
print(f"TimesNet (amplitude-wtd): Test Acc = {logs_aw['test_acc'][-1]:.4f}")
