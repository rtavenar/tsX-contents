# %% [markdown]
# # Basics of time series forecasting
# 
# This lab introduces the fundamentals of time series forecasting with deep learning. 
# You will learn how to build data loaders for time series data, implement simple 
# autoregressive models, and experiment with preprocessing techniques such as 
# standardization and differencing. The lab covers training and evaluation procedures, 
# and compares the impact of different modeling choices on forecasting performance.

# %% 
import torch
import numpy
import matplotlib.pyplot as plt

# %% [markdown]
# ## Part 1: Data loaders
# 
# In this section, you will load a standard time series forecasting dataset and prepare a data loader for it.
#
# To begin, visit <https://github.com/zhouhaoyi/ETDataset> and download the ETTh1 dataset as a CSV file.
#
# **Question 1.** Visualize the dataset, focusing on the univariate time series corresponding to the target variable. Do you observe a trend? Periodicity? Any abnormal segments?

# %% + tags=["solution"]
raw = numpy.loadtxt("data/ETTh1.csv", delimiter=",", skiprows=1, usecols=-1)
series = raw.astype(numpy.float32)
plt.plot(series, color='blue')
plt.xlabel("Time (hours)")
plt.ylabel("Oil Temperature (Celsius degrees)")
plt.show()

# %% + tags=["solution"]
plt.plot(series[:500], color='blue')
plt.xlabel("Time (hours)")
plt.ylabel("Oil Temperature (Celsius degrees)")
plt.show()


# %% [markdown]
# **Question 2.** Implement a PyTorch `DataLoader` that reads the CSV file at initialization time, allows you to specify the past window length and forecast horizon, and provides batches of `(past, horizon)` pairs for the univariate forecasting problem.

# %% + tags=["solution"]
class ForecastingDataset(torch.utils.data.Dataset):
    """Windowed univariate forecasting dataset."""

    def __init__(self, 
                 csv_path: str, 
                 window: int, 
                 horizon: int, 
                 target_col: int = -1):
        super().__init__()
        raw = numpy.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=target_col)
        series = raw.astype(numpy.float32)
        self.window = window
        self.horizon = horizon
        self.series = series
        self.max_start = len(series) - window - horizon + 1
        if self.max_start < 1:
            raise ValueError("Window + horizon larger than available series length")

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx: int):
        start = idx
        past = self.series[start : start + self.window]
        future = self.series[start + self.window : start + self.window + self.horizon]
        past = torch.from_numpy(past)  # shape: (window,)
        future = torch.from_numpy(future)  # shape: (horizon,)
        return past, future


def build_dataloader(csv_path: str, 
                     window: int, 
                     horizon: int, 
                     batch_size: int = 32, 
                     shuffle: bool = True):
    """Create a DataLoader emitting `(past, horizon)` batches."""
    dataset = ForecastingDataset(csv_path=csv_path, 
                                 window=window, 
                                 horizon=horizon)
    return torch.utils.data.DataLoader(dataset, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle, 
                                       drop_last=False)

# %% [markdown] 
# **Question 3.** Improve your `build_dataloader` function above to build both a training data loader and a validation data loader. What would be appropriate choices for a clean separation between training and validation datasets?

# %% + tags=["solution"]
class ForecastingDataset(torch.utils.data.Dataset):
    """Windowed univariate forecasting dataset."""

    def __init__(self, 
                 csv_path: str, 
                 window: int, 
                 horizon: int, 
                 target_col: int = -1,
                 start: int = 0,
                 end: int = None):
        super().__init__()
        raw = numpy.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=target_col)
        series = raw.astype(numpy.float32)
        if end is None:
            series = series[start:]
        else:
            series = series[start:end]
        self.window = window
        self.horizon = horizon
        self.series = series
        self.max_start = len(series) - window - horizon + 1
        if self.max_start < 1:
            raise ValueError("Window + horizon larger than available series length")

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx: int):
        start = idx
        past = self.series[start : start + self.window]
        future = self.series[start + self.window : start + self.window + self.horizon]
        past = torch.from_numpy(past)  # shape: (window,)
        future = torch.from_numpy(future)  # shape: (horizon,)
        return past, future


def build_dataloader(csv_path: str, 
                     window: int, 
                     horizon: int, 
                     batch_size: int = 32, 
                     shuffle: bool = True):
    """Create a DataLoader emitting `(past, horizon)` batches."""
    dataset = ForecastingDataset(csv_path=csv_path, 
                                 window=window, 
                                 horizon=horizon)
    train_dataset = ForecastingDataset(csv_path=csv_path, 
                                       window=window, 
                                       horizon=horizon,
                                       end=len(dataset) // 5)
    train_dl = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=shuffle, 
                                           drop_last=False)
    valid_dataset = ForecastingDataset(csv_path=csv_path, 
                                       window=window, 
                                       horizon=horizon,
                                       start=len(dataset) // 5)
    valid_dl = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=batch_size, 
                                           shuffle=shuffle, 
                                           drop_last=False)
    return train_dl, valid_dl


# %% [markdown]
# ## Part 2: First models
#
# In this part, you will build your first few forecasting models, train them, and
# experiment with classical detrending techniques used in time series analysis.

# %% [markdown]
# **Question 4.** Implement a simple autoregressive (AR) model in `torch`.
#
# The model should:
# - take a past window of shape `(batch, window)`
# - output a forecast of shape `(batch, horizon)`
# - be linear in the inputs

# %% + tags=["solution"]
class ARModel(torch.nn.Module):
    """Linear autoregressive model."""

    def __init__(self, window: int, horizon: int):
        super().__init__()
        self.linear = torch.nn.Linear(window, horizon)

    def forward(self, past):
        # past: (batch, window)
        # output: (batch, horizon)
        return self.linear(past)

# %% [markdown]
# **Question 5.** Train the AR model using mean squared error (MSE) and evaluate
# it on the validation set.
#
# Implement:
# - a training loop
# - a validation loop
# - reporting of train and validation losses

# %% + tags=["solution"]
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for past, future in dataloader:
        optimizer.zero_grad()
        pred = model(past)
        loss = criterion(pred, future)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * past.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    for past, future in dataloader:
        pred = model(past)
        loss = criterion(pred, future)
        total_loss += loss.item() * past.size(0)

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

# Example usage
window = 96
horizon = 24
n_epochs = 50

train_dl, valid_dl = build_dataloader(
    csv_path="data/ETTh1.csv",
    window=window,
    horizon=horizon,
    batch_size=64
)

ar_model = ARModel(window, horizon)
optimizer = torch.optim.Adam(ar_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
logs_AR = train_and_valid_loop(ar_model, train_dl, valid_dl, optimizer, criterion, n_epochs)

# %% [markdown]
# **Question 6.** Add input/output standardization layers.

# %% + tags=["solution"]
class StandardScaler(torch.nn.Module):
    """Fixed standardization layer."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-6)

    def inverse(self, x):
        return x * (self.std + 1e-6) + self.mean

def compute_scaler(dataloader):
    values = []
    for past, future in dataloader:
        values.append(past[:, :1])
    values = torch.cat(values, dim=0)
    mean = values.mean()
    std = values.std()
    return mean, std


class ScaledARModel(ARModel):
    def __init__(self, window, horizon, mean, std):
        super().__init__(window, horizon)
        self.scaler = StandardScaler(mean, std)

    def forward(self, past):
        past_scaled = self.scaler(past)
        pred_scaled = self.linear(past_scaled)
        return self.scaler.inverse(pred_scaled)

# Compute scaler from training data only
mean, std = compute_scaler(train_dl)
scaled_ar_model = ScaledARModel(window, horizon, mean, std)
optimizer = torch.optim.Adam(scaled_ar_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
logs_scaledAR = train_and_valid_loop(scaled_ar_model, train_dl, valid_dl, optimizer, criterion, n_epochs)

# %% [markdown]
# **Question 7.** Implement a differencing layer that removes local trends:
#
#   $x_t' = x_t - x_{t-1}$
#
# Then implement the inverse operation (integration) to recover forecasts
# back to the original scale.
#
# Experiment with how differencing affects convergence and final error, and 
# compare forecasts qualitatively.

# %% + tags=["solution"]
class Differencing(torch.nn.Module):
    def forward(self, x):
        # x: (batch, window)
        return x[:, 1:] - x[:, :-1]


class Integration(torch.nn.Module):
    def forward(self, last_value, diffs):
        # last_value: (batch, 1)
        # diffs: (batch, horizon)
        return last_value + torch.cumsum(diffs, dim=1)

class DifferencedScaledARModel(torch.nn.Module):
    def __init__(self, window, horizon, mean, std):
        super().__init__()
        self.scaler = StandardScaler(mean, std)
        self.diff = Differencing()
        self.ar = ARModel(window - 1, horizon)
        self.integrate = Integration()

    def forward(self, past):
        scaled_past = self.scaler(past)
        last_value = scaled_past[:, -1:].detach()
        diffs = self.diff(scaled_past)
        pred_diffs = self.ar(diffs)
        scaled_preds = self.integrate(last_value, pred_diffs)
        return self.scaler.inverse(scaled_preds)

# Compute scaler from training data only
mean, std = compute_scaler(train_dl)
diff_scaled_ar_model = DifferencedScaledARModel(window, horizon, mean, std)
optimizer = torch.optim.Adam(diff_scaled_ar_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

logs_DifferencedScaledAR = train_and_valid_loop(diff_scaled_ar_model, train_dl, valid_dl, optimizer, criterion, n_epochs)

# %% + tags=["solution"]
def viz_forecast(list_models: list, dataloader: torch.utils.data.DataLoader, ts_id: int = 0):
    past_val, future_val = next(iter(dataloader))
    plt.plot(numpy.arange(past_val.shape[1]),
            past_val[ts_id],
            label="Past window"
    )
    plt.plot(past_val.shape[1] + numpy.arange(future_val.shape[1]),
            future_val[ts_id],
            label="Ground truth horizon"
    )
    for model in list_models:
        pred_val = model(past_val).detach().numpy()
        plt.plot(past_val.shape[1] + numpy.arange(pred_val.shape[1]),
                pred_val[ts_id],
                label=f"Forecast ({model.__class__.__name__})"
        )
    plt.legend()
    plt.show()

viz_forecast([diff_scaled_ar_model, scaled_ar_model], valid_dl)

# %% [markdown]
# **Question 8.** Replace the linear layer in your best-performing AR model with MLPs.
#
# Compare:
# - convergence speed
# - validation error

# %% + tags=["solution"]
class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class DifferencedScaledDeepARModel(DifferencedScaledARModel):
    def __init__(self, window, horizon, mean, std):
        super().__init__(window, horizon, mean, std)
        self.ar = MLP(window - 1, horizon)

# Compute scaler from training data only
mean, std = compute_scaler(train_dl)
diff_scaled_deep_ar_model = DifferencedScaledDeepARModel(window, horizon, mean, std)
optimizer = torch.optim.Adam(diff_scaled_deep_ar_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

logs_DifferencedScaledDeepAR = train_and_valid_loop(diff_scaled_deep_ar_model, train_dl, valid_dl, optimizer, criterion, n_epochs)


# %% [markdown]
# **Wrap-up question.** Which factor had the largest impact on performance?
# 1. model complexity
# 2. scaling
# 3. differencing
#
# Justify your answer with quantitative results and/or plots.

# %% + tags=["solution"]
plt.plot(logs_AR["valid_loss"], label="AR")
plt.plot(logs_scaledAR["valid_loss"], label="Scaled AR")
plt.plot(logs_DifferencedScaledAR["valid_loss"], label="Differenced + Scaled AR")
plt.plot(logs_DifferencedScaledDeepAR["valid_loss"], label="Differenced + Scaled Deep AR")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.show()

# %% [markdown]
# **Bonus question.** Given what you know about this time series, do you expect seasonality? 
# Implement an additional layer for your scaled differenced AR model that performs seasonal differencing using a lag equal to the expected period $\Delta_t$:
#
# $x_t' = x_t - x_{t-\Delta_t}$
#
# Does it help improve overall performance?

# %% + tags=["solution"]
class SeasonalDifferencing(torch.nn.Module):
    """Seasonal differencing: x_t' = x_t - x_{t-lag}"""
    
    def __init__(self, lag: int):
        super().__init__()
        self.lag = lag
    
    def forward(self, x):
        return x[:, self.lag:] - x[:, :-self.lag]


class SeasonalIntegration(torch.nn.Module):
    """Inverse of seasonal differencing: x_t = x_{t-lag} + x_t'"""
    
    def __init__(self, lag: int):
        super().__init__()
        self.lag = lag
    
    def forward(self, last_season_values, diff_preds):
        horizon = diff_preds.shape[1]
        preds = []
        extended_history = last_season_values  # (batch, lag)
        
        for t in range(horizon):
            # The base value is from 'lag' steps ago
            if t < self.lag:
                # Use value from last_season_values
                base_value = extended_history[:, t:t+1]
            else:
                # Use previously predicted value
                base_value = preds[t - self.lag]
            
            # Compute prediction: x_t = x_{t-lag} + diff
            new_pred = base_value + diff_preds[:, t:t+1]
            preds.append(new_pred)
        
        return torch.cat(preds, dim=1)


class SeasonalDifferencedScaledARModel(torch.nn.Module):
    """AR model with seasonal differencing and scaling"""
    
    def __init__(self, window: int, horizon: int, mean: float, std: float, lag: int = 24):
        super().__init__()
        assert(lag < window)
        self.scaler = StandardScaler(mean, std)
        self.seasonal_diff = SeasonalDifferencing(lag=lag)
        self.ar = ARModel(window - lag, horizon)  # Window size reduced by lag
        self.seasonal_integrate = SeasonalIntegration(lag=lag)
        self.lag = lag
    
    def forward(self, past):
        scaled_past = self.scaler(past)
        last_season_values = scaled_past[:, -self.lag:].detach() # Save for later (integration)
        seasonally_differenced = self.seasonal_diff(scaled_past)
        pred_diffs = self.ar(seasonally_differenced)
        scaled_preds = self.seasonal_integrate(last_season_values, pred_diffs)
        return self.scaler.inverse(scaled_preds)

# Assuming daily seasonality for hourly data (lag=24 hours)
mean, std = compute_scaler(train_dl)
seasonal_diff_scaled_ar_model = SeasonalDifferencedScaledARModel(
    window=window, 
    horizon=horizon, 
    mean=mean, 
    std=std, 
    lag=24  # Daily seasonality for hourly data
)
optimizer = torch.optim.Adam(seasonal_diff_scaled_ar_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
logs_SeasonalDifferencedScaledAR = train_and_valid_loop(
    seasonal_diff_scaled_ar_model, train_dl, valid_dl, optimizer, criterion, n_epochs
)