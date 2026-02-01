# %% [markdown]
# # Time Series Foundation Models
#
# In this lab, you will compare foundation-model-based approaches for time series
# classification and forecasting. You will:
# - Use **MANTIS** (a lightweight TSC foundation model) and **MultiROCKET** (a fast
#   feature-based baseline) on a classification task
# - Use **TiReX** for probabilistic forecasting on the ETTh dataset and visualize
#   predictions with uncertainty intervals
#
# Data loaders and setup steps are provided below.

# %% [markdown]
# ## Part 0: Setup
#
# **Question 0.** Install the required packages: `mantis-tsfm` (MANTIS), `aeon`
# (MultiROCKET and data utilities), and `tirex-ts` (TiReX forecaster).

# %%
!pip install mantis-tsfm aeon tirex-ts

# %% [markdown]
# ## Part 1: Time Series Classification
#
# In this part you will compare MANTIS and MultiROCKET on the UCI HAR dataset,
# then optionally combine their features.
#
# **Question 1.** Download the UCI HAR classification dataset (if needed) and
# prepare the train/test arrays. Use the code below so that `X_train`, `y_train`,
# `X_test`, `y_test` are available with shapes suitable for the classifiers
# (e.g. `X_*` of shape `(n_samples, n_channels, n_timesteps)` for aeon/MANTIS).

# %%
!wget -O data/UCIHAR.npz https://github.com/rtavenar/ml-datasets/releases/download/UCIHAR/UCIHAR.npz

# %%
import numpy as np

dataset = np.load("data/UCIHAR.npz")

X_train, y_train = dataset["X_train"], dataset["y_train"]
X_test, y_test = dataset["X_test"], dataset["y_test"]

X_train.shape, X_test.shape

# %%
X_train = X_train.swapaxes(1, 2)
y_train = y_train.ravel()
X_test = X_test.swapaxes(1, 2)
y_test = y_test.ravel()

# %% [markdown]
# **Question 2.** Use **MANTIS** from the `mantis-tsfm` module: load the pre-trained
# Mantis8M model, extract embeddings on the training and test sets with `transform`,
# then train a linear classifier (e.g. logistic regression) on the extracted features
# and report test accuracy.

# %%
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer

network = Mantis8M(device="cpu")
network = network.from_pretrained("paris-noah/Mantis-8M")

model_mantis = MantisTrainer(network=network, device="cpu")
# model_mantis.fit(X_train, y_train.flatten(), num_epochs=10, fine_tuning_type="head")
# y_pred = model_mantis.predict(X_test)
Z_train_mantis = model_mantis.transform(X_train)
Z_test_mantis = model_mantis.transform(X_test)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(l1_ratio=1., random_state=0).fit(Z_train_mantis, y_train)
clf.score(Z_test_mantis, y_test)

# %% [markdown]
# **Question 3.** Use **MultiROCKET** from the `aeon` library: fit the MultiRocket
# transformer on the training set, extract features on train and test, train a
# linear classifier on these features, and report test accuracy. Compare this
# accuracy with the one obtained with MANTIS in Question 2.

# %%
from aeon.transformations.collection.convolution_based import MultiRocket

model_mr = MultiRocket(n_kernels=512)
Z_train_mr = model_mr.fit_transform(X_train)
clf = LogisticRegression(l1_ratio=1., random_state=0).fit(Z_train_mr, y_train)

Z_test_mr = model_mr.transform(X_test)
clf.score(Z_test_mr, y_test)

# %% [markdown]
# **Question 4.** Concatenate the MANTIS and MultiROCKET feature vectors and train a linear classifier on the
# combined features. Does combining both representations improve test accuracy
# compared to using either one alone?

# %%
Z_train_concat = np.concatenate((Z_train_mantis, Z_train_mr), axis=1)
Z_test_concat = np.concatenate((Z_test_mantis, Z_test_mr), axis=1)
clf = LogisticRegression(random_state=0).fit(Z_train_concat, y_train)
clf.score(Z_test_concat, y_test)


# %% [markdown]
# ## Part 2: Probabilistic Forecasting with TiReX
#
# In this part you will use the **TiReX** foundation model for time series
# forecasting on the univariate ETTh1 series and visualize probabilistic
# predictions.
#
# **Question 5.** Build a windowed forecasting dataset and DataLoader for ETTh1
# (e.g. context length 96, horizon 96). Load the pre-trained TiReX model from
# the `tirex-ts` module and produce probabilistic forecasts (e.g. mean and
# quantiles) on a batch of windows. Plot the past observations, ground-truth
# future, mean prediction, and uncertainty interval (e.g. 90%) for a few samples.

# %%
import torch
class ForecastingDataset(torch.utils.data.Dataset):
    """Windowed univariate forecasting dataset."""

    def __init__(self, 
                 csv_path: str, 
                 window: int, 
                 horizon: int, 
                 target_col: int = -1):
        super().__init__()
        raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=target_col)
        series = raw.astype(np.float32)
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
dataloader = build_dataloader("data/ETTh1.csv", window=96, horizon=96)

# %%
from tirex import load_model, ForecastModel
import matplotlib.pyplot as plt

model: ForecastModel = load_model("NX-AI/TiRex")

def visualize_probabilistic_forecast(model, dataloader, n_samples=3):
    """Visualize probabilistic forecasts with uncertainty intervals."""
    model.eval()
    past, future = next(iter(dataloader))
    with torch.no_grad():
        quantiles, mean = model.forecast(past, prediction_length=future.shape[1])
        print(quantiles.shape, mean.shape)
    
    for i in range(min(n_samples, past.shape[0])):
        # Plot past
        t_past = np.arange(past.shape[1])
        plt.plot(t_past, past[i].numpy(), 'b-', linewidth=2, label='Past observations')
        
        # Plot future
        t_future = np.arange(past.shape[1], past.shape[1] + future.shape[1])
        plt.plot(t_future, future[i].numpy(), 'g-', linewidth=2, label='Ground truth')
        
        # Plot mean prediction
        plt.plot(t_future, mean[i].numpy(), 'r--', linewidth=2, label='Mean prediction')
        
        # Plot uncertainty bands
        plt.fill_between(t_future, quantiles[i, :, 0].numpy(), quantiles[i, :, -1].numpy(), alpha=0.2, color='red', label='80% interval')
        
        plt.axvline(x=past.shape[1]-0.5, color='gray', linestyle=':', alpha=0.5)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Probabilistic Forecast (Sample {i+1})')
        plt.show()

visualize_probabilistic_forecast(model, dataloader, n_samples=3)