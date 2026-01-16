# %% [markdown]
# # RNNs for Time Series Forecasting
# 
# In this lab, you will explore recurrent models (RNN, GRU, LSTM, xLSTM) and
# temporal convolutions (CNN/TCN) for forecasting. You will work on two tasks:
# 1) a synthetic long-range-dependency dataset, and 2) the multivariate ETTh1 dataset.

# %%
import math
import numpy
import torch
import matplotlib.pyplot as plt

# %% [markdown]
# ## Part 1 — Toy data for long-term memory experiments
# 
# We build a synthetic sequence where a short spike far in the past determines a
# pattern in the future (e.g., a sine burst at a fixed lag). This stresses
# long-range credit assignment.

# %% [markdown]
# The dataset below is composed of sequences with:
# - a base noisy sine
# - a unique spike placed uniformly in the first half
# - a sine is located in the target at a fixed lag 
#   after the spike occurred in the base

# %%
class SpikeLagDataset(torch.utils.data.Dataset):
    """Synthetic long-range-dependency dataset."""

    def __init__(self, length=200, lag=80, n_samples=5000, noise=0.05):
        self.length = length
        self.lag = lag
        self.n_samples = n_samples
        self.noise = noise

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = torch.linspace(0, 4 * math.pi, self.length)
        base = torch.sin(t) + self.noise * torch.randn_like(t)
        spike_pos = torch.randint(10, self.length // 2, (1,)).item()
        spike = torch.zeros_like(base)
        spike[spike_pos] = 3.0
        series = base + spike
        target = torch.zeros(self.length)
        target[spike_pos + self.length // 4 :spike_pos + self.length // 2] = torch.sin(
            t[:self.length // 4]
        )
        return series.unsqueeze(-1), target.unsqueeze(-1)

def build_spike_dataloaders(length=200, lag=80, batch_size=64, n_train=4000, n_valid=500):
    train_ds = SpikeLagDataset(length=length, lag=lag, n_samples=n_train)
    valid_ds = SpikeLagDataset(length=length, lag=lag, n_samples=n_valid)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    return train_dl, valid_dl
train_dl, valid_dl = build_spike_dataloaders()

# %% [markdown]
# **Question.** Visualize a few series from this dataset.

# %% + tags=["solution"]
def plot_spike_samples(dataset, n=3):
    fig, axs = plt.subplots(n, 1, figsize=(8, 6))
    for i in range(n):
        x, y = dataset[i]
        axs[i].plot(range(len(x)), x.numpy(), label="input")
        axs[i].plot(range(len(x), len(x) + len(y)), y.numpy(), label="target")
        axs[i].legend()
    plt.show()

length, lag, n_train = 200, 80, 10
dataset = SpikeLagDataset(length=length, lag=lag, n_samples=n_train)
plot_spike_samples(dataset, n_train)

# %% [markdown]
# **Question.** Implement a simple RNN forecaster:
# - Use `nn.RNN`, process the full sequence, and map the final hidden state to
#   the target segment with a linear head.
# - Report train/valid loss (MSE).

# %% [python]
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        pred = model(x)  # pred: (batch, horizon, 1)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    for x, y in dataloader:
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def train_and_valid_loop(model, train_dl, valid_dl, optimizer, criterion, n_epochs):
    logs = {"train_loss": [], "valid_loss": []}
    print(model.__class__.__name__)
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_dl, optimizer, criterion)
        logs["train_loss"].append(train_loss)
        valid_loss = eval_epoch(model, valid_dl, criterion)
        logs["valid_loss"].append(valid_loss)
        print(f"Epoch {epoch:02d} | train={train_loss:.4f} | valid={valid_loss:.4f}")
    return logs

# %% + tags=["solution"]
class RNNForecaster(torch.nn.Module):
    def __init__(self, input_dim=1, hidden=64, horizon=200):
        super().__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden, batch_first=True)
        self.horizon = horizon
        self.head = torch.nn.Linear(hidden, horizon)

    def forward(self, x):
        out, h = self.rnn(x)  # h: (1, batch, hidden)
        h_last = h[0]
        pred = self.head(h_last)
        return pred.unsqueeze(-1)  # (batch, horizon, 1)

hidden_dim = 64
rnn = RNNForecaster(hidden=hidden_dim, horizon=length)
opt = torch.optim.Adam(rnn.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
logs_rnn = train_and_valid_loop(rnn, train_dl, valid_dl, opt, criterion, n_epochs=20)

# %% [markdown]
# **Question.** Replace the RNN with GRU and LSTM. Compare:
# - convergence speed and final validation loss
# - qualitative forecasts

# %% + tags=["solution"]
class GRUForecaster(RNNForecaster):
    def __init__(self, input_dim=1, hidden=64, horizon=200):
        super().__init__(input_dim, hidden, horizon)
        self.rnn = torch.nn.GRU(input_dim, hidden, batch_first=True)


class LSTMForecaster(RNNForecaster):
    def __init__(self, input_dim=1, hidden=64, horizon=200):
        super().__init__(input_dim, hidden, horizon)
        self.rnn = torch.nn.LSTM(input_dim, hidden, batch_first=True)

    def forward(self, x):
        out, (h, c) = self.rnn(x)  # h: (1, batch, hidden)
        h_last = h[0]
        pred = self.head(h_last)
        return pred.unsqueeze(-1)  # (batch, horizon, 1)

hidden_dim = 64
gru = GRUForecaster(hidden=hidden_dim, horizon=length)
opt = torch.optim.Adam(gru.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
logs_gru = train_and_valid_loop(gru, train_dl, valid_dl, opt, criterion, n_epochs=20)

hidden_dim = 64
lstm = LSTMForecaster(hidden=hidden_dim, horizon=length)
opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
logs_lstm = train_and_valid_loop(lstm, train_dl, valid_dl, opt, criterion, n_epochs=20)

# %% [markdown]
# The code below provides implementation for a TCN block.

# %% [python]
from torch.nn.utils.parametrizations import weight_norm

class CausalConv(torch.nn.Module):
    """Minimal 1D causal convolution block without activation."""

    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = weight_norm(torch.nn.Conv1d(in_ch, out_ch, k, dilation=d))

    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)

class TCNBlock(torch.nn.Module):
    """A TCN block involving 2 dilated convolutions."""

    def __init__(self, in_ch, out_ch, k=3, d1=1, d2=2, dropout_rate=.1):
        super().__init__()
        self.conv1 = CausalConv(in_ch, out_ch, k, d1)
        self.conv2 = CausalConv(out_ch, out_ch, k, d2)
        # 1x1 conv to project residual to the correct number of channels
        self.conv_1x1 = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.do1 = torch.nn.Dropout1d(p=dropout_rate)
        self.do2 = torch.nn.Dropout1d(p=dropout_rate)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        residual = self.conv_1x1(x)
        out = self.conv1(x)
        out = self.act(out)
        out = self.do1(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.do2(out)
        return self.act(out + residual)  # (batch, channels, time)

# %% [markdown]
# **Question.** Based on the above implementation, try the following TCN baselines
# for your forecasting problem:
# - 1D causal CNN with dilations (a 1-block TCN).
# - Stack several TCN blocks such that the receptive field is sufficient for the task at stake.
# Compare against RNN/GRU/LSTM on the spike-lag task.


# %% + tags=["solution"]

class TCNForecaster(torch.nn.Module):
    def __init__(self, input_dim=1, hidden=64, n_blocks=3, k=3, horizon=200):
        super().__init__()
        blocks = []
        ch_in = input_dim
        for i in range(n_blocks):
            blocks.append(TCNBlock(ch_in, hidden, k, d1=2 ** (2*i), d2=2 ** (2*i+1)))
            ch_in = hidden
        self.net = torch.nn.Sequential(*blocks)
        self.head = torch.nn.Linear(hidden, horizon)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, channels, time)
        feats = self.net(x)
        last_feat = feats[:, :, -1]
        pred = self.head(last_feat)
        return pred.unsqueeze(-1)

hidden_dim = 64
tcn = TCNForecaster(hidden=hidden_dim, horizon=length)
opt = torch.optim.Adam(tcn.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
logs_tcn = train_and_valid_loop(tcn, train_dl, valid_dl, opt, criterion, n_epochs=20)

dict_logs = {
    "TCN": logs_tcn,
    "RNN": logs_rnn,
    "LSTM": logs_lstm,
    "GRU": logs_gru
}
for label, logs in dict_logs.items():
    plt.plot(logs["valid_loss"], label=label)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.show()


dict_models = {
    "TCN": tcn,
    "RNN": rnn,
    "LSTM": lstm,
    "GRU": gru
}
for i in range(3):
    past, future = dataset[i]
    plt.figure()
    plt.plot(range(length), past.numpy())
    plt.plot(range(length, 2*length), future.numpy(), label="Ground truth")
    for label, model in dict_models.items():
        pred = model(past.unsqueeze(0))[0]
        plt.plot(range(length, 2*length), pred.flatten().detach().numpy(), label=label)
    plt.legend()
    plt.show()

# %% [markdown]
# ## Part 2 — Multivariate ETTh1 forecasting
# 
# We now revisit ETTh1 used in the previous lab session but now
# treat it as a multivariate-to-univariate 
# forecasting problem.

# %% [markdown]
# Below is an adaptation from last session's `ETTh1Dataset` class in which we:
# - Allow specifying `window`, `horizon`, and optional time-of-day encoding
#   (scaled to [0, 1]).
# - Return tensors shaped `(batch, time, features)` where features>1 for input 
#   tensors and features=1 for output tensor (target).

# %% [python]
def load_etth1(csv_path, use_time_feat=True):
    def to_str(str_or_bytes):
        if isinstance(str, str_or_bytes):
            return str_or_bytes
        else:
            return str_or_bytes.decode()
    
    d_conv = {
        0: (lambda x: float(to_str(x).split(" ")[1].split(":")[0]))
    }
    raw = numpy.loadtxt(csv_path, delimiter=",", skiprows=1, converters=d_conv)
    features = raw.astype(numpy.float32)
    if use_time_feat:
        features[:, 0] /= 23.
    else:
        features = features[:, 1:]
    return features


class ETTh1Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, window, horizon, use_time_feat=True, start=0, end=None, mean=None, std=None):
        feats = load_etth1(csv_path, use_time_feat=use_time_feat)
        feats = feats[start:]
        if end is not None:
            feats = feats[:end]
        self.feats = feats
        self.window = window
        self.horizon = horizon
        self.max_start = len(feats) - window - horizon + 1
        if self.max_start < 1:
            raise ValueError("Window + horizon exceeds series length")
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        past = self.feats[idx : idx + self.window]
        future = self.feats[idx + self.window : idx + self.window + self.horizon, -1:]
        # Apply scaling if mean and std are provided
        if self.mean is not None and self.std is not None:
            past = (past - self.mean) / self.std
            future = (future - self.mean[-1:]) / self.std[-1:]
        return torch.from_numpy(past), torch.from_numpy(future)

# %% [markdown]
# **Question.** Based on the `ETTh1Dataset` class provided above, implement 
# a `build_etth1_dataloaders` function that generates training and validation 
# data loaders, with scaling. Load the data relying on a past window of size 96 
# and a horizon of 24 time steps.

# %% + tags=["solution"]
def build_etth1_dataloaders(csv_path, window=96, horizon=24, batch_size=64, split=0.8):
    dataset = ETTh1Dataset(csv_path, window, horizon)
    n = len(dataset)
    n_train = int(split * n)
    train_ds = ETTh1Dataset(csv_path, window, horizon, end=n_train)
    mean = numpy.mean(train_ds.feats, axis=0)  # (n_features,)
    std = numpy.std(train_ds.feats, axis=0)  # (n_features,)
    std = numpy.where(std < 1e-8, 1.0, std)
    
    # Create datasets with scaling
    train_ds = ETTh1Dataset(csv_path, window, horizon, end=n_train, mean=mean, std=std)
    valid_ds = ETTh1Dataset(csv_path, window, horizon, start=n_train, mean=mean, std=std)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    return train_dl, valid_dl

window = 96
horizon = 24
dataset = ETTh1Dataset("data/ETTh1.csv", window=window, horizon=horizon)
train_dl, valid_dl = build_etth1_dataloaders("data/ETTh1.csv", window=window, horizon=horizon)

# %% [markdown]
# **Question.** Re-use TCN and GRU forecasting models from part 1 and 
# evaluate their performance on this dataset. 
# Does the TCN's receptive field cover the whole input window in this case?

# %% + tags=["solution"]
hidden_dim = 64
channels = 8
horizon = 24
gru = GRUForecaster(input_dim=channels, hidden=hidden_dim, horizon=horizon)
opt = torch.optim.Adam(gru.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
logs_gru = train_and_valid_loop(gru, train_dl, valid_dl, opt, criterion, n_epochs=20)

hidden_dim = 64
tcn = TCNForecaster(input_dim=channels, hidden=hidden_dim, horizon=horizon)
opt = torch.optim.Adam(tcn.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
logs_tcn = train_and_valid_loop(tcn, train_dl, valid_dl, opt, criterion, n_epochs=20)

dict_logs = {
    "TCN": logs_tcn,
    "GRU": logs_gru
}
for label, logs in dict_logs.items():
    plt.plot(logs["valid_loss"], label=label)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.show()

# %%
dict_models = {
    "TCN": tcn,
    "GRU": gru
}
it = iter(valid_dl)
for i in range(3):
    past, future = next(it)
    plt.figure()
    plt.plot(range(window), past[0, :, -1].numpy())
    plt.plot(range(window, window + horizon), future[0].numpy(), label="Ground truth")
    for label, model in dict_models.items():
        pred = model(past)[0]
        plt.plot(range(window, window + horizon), pred.flatten().detach().numpy(), label=label)
    plt.legend()
    plt.show()

# %% [markdown]
# **Question.** Now you will use the code from the `xlstm` package to build an xLSTM
# model for this forecasting task. What stack of sLSTM/mLSTM blocks is used here?

# %%
%pip install xlstm

# %% [python]
from xlstm.xlstm_block_stack import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig

class xLSTMForecaster(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        window: int,
        horizon: int,
    ):
        super().__init__()

        # Project input channels → model dimension
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)

        # Build block configs (repeat to get depth)
        layer_config = mLSTMLayerConfig(
            embedding_dim=hidden_dim,
            num_heads=2,
        )
        block_cfg = mLSTMBlockConfig(
            mlstm=layer_config,
        )

        xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=block_cfg,
            slstm_at=[],
            num_blocks=num_blocks,
            context_length=window,
            embedding_dim=hidden_dim
        )

        self.xlstm = xLSTMBlockStack(xlstm_cfg)

        # Forecast head
        self.head = torch.nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        x = self.input_proj(x)        # (B, T, c)
        y = self.xlstm(x)             # (B, T, c)

        last_state = y[:, -1]         # (B, c)
        out = self.head(last_state)   # (B, H)

        return out.unsqueeze(-1)      # (B, H, 1)

# %% [markdown]
# **Question.** Train your own xLSTM forecaster and check its performance.

# %% + tags=["solution"]
xlstm_model = xLSTMForecaster(
    input_dim=channels,
    hidden_dim=hidden_dim,
    num_blocks=1,
    horizon=horizon,
    window=window
)
opt = torch.optim.Adam(xlstm_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
logs_xlstm = train_and_valid_loop(xlstm_model, train_dl, valid_dl, opt, criterion, n_epochs=20)