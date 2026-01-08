#import "@preview/touying:0.6.1": *
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning for Time Series - Basics],
  config-info(
    title: [Deep Learning for Time Series],
    subtitle: [Session 1: Basics],
    author: [Romain Tavenard],
    date: []
  ),
  config-colors(
    primary: rgb(131,109,169),
    secondary: rgb(200,200,200),
    tertiary: rgb(200,200,200),
    primary-light: rgb(200,200,200)
  )
)

#title-slide()

== Definition

In this course, we will denote by _time series_ any sequence of feature vectors:
$
 X = (x_t)_(t=1)^T " such that " x_t in RR^d
$

#align(center)[
  #include "cetz/time_series.typ"
]

= Time series tasks

== Classification / labeling

- *Labeling*
  - Assign a class label to each time point (or segment)
  #align(center)[
    #include "cetz/labeling.typ"
  ]
- *Classification*
  - Assign a class label to a full series
  #align(center)[
    #include "cetz/classification.typ"
  ]

== Anomaly detection

- Find unexpected points or segments
  - Usually unsupervised

  #align(center)[
    #include "cetz/anomaly_detection.typ"
  ]

== Change-point detection

- Detect structural breaks in the sequence

  #align(center)[
    #include "cetz/change_point_detection.typ"
  ]

== Forecasting

- Regression task
- Target values are called *horizons*
- Model should capture temporal dependencies:

  $ 
  X_"horizon" approx f(X_"past")
  $

  #align(center)[
    #include "cetz/forecasting.typ"
  ]

  - Modern pipelines: forecasting is *the* pretext task for feature learning

= Statistical models

== Additive decomposition

- Typical (old-school) assumption: Time series can be decomposed into: Trend, Seasonality, and Residuals in an additive way:

  $
  x_t = T_t + S_t + R_t
  $

== ARIMA models

- AR, MA, ARMA, ARIMA

= Evaluation metrics and losses

== Metrics

- MAE, RMSE, MASE, sMAPE

== Alignment-based losses

- softDTW, TDI, DILATE

// - Objective: Understand fundamental concepts and common tasks.
// - Plan: definitions → tasks → focus on forecasting → classical non-deep models.

// == What is a time series?

// - A sequence of observations indexed by time: $x_t$ for $t=1,2,dots$.
// - Sampling: discrete (e.g., daily, hourly) or continuous (often discretized).
// - Key properties: trend, seasonality, noise, stationarity.

// == Typical time-series tasks

// - Forecasting: estimate $x_{t+h}$ for $h>0$.
// - Anomaly detection: find unexpected points or segments.
// - Classification / labeling: assign a class to a series or segment.
// - Segmentation / change-point detection: detect structural breaks.
// - Imputation: fill missing values.

// == Why focus on forecasting?

// - Widely used in practice (finance, energy, logistics, health).
// - Often used as a pretext task in modern pipelines: pretraining, quick evaluation, feature generation.
// - Easy to compare classical models and ML/DL approaches on forecasting tasks.

// == Classical models — AR / MA / ARMA / ARIMA

// **Autoregressive AR(p)**

// - Form: $x_t = c + sum_{i=1}^p phi_i x_{t-i} + epsilon_t$.
// - Captures temporal correlations via lags.

// **Moving Average MA(q)**

// - Form: $x_t = mu + epsilon_t + sum_{j=1}^q theta_j epsilon_{t-j}$.
// - Models the effect of past shocks (noise) on the current value.

// **ARMA(p,q)**

// - $x_t = c + sum_{i=1}^p phi_i x_{t-i} + epsilon_t + sum_{j=1}^q theta_j epsilon_{t-j}$.
// - Requires stationarity.

// **ARIMA(p,d,q)**

// - Introduces differencing of order $d$ to handle non-stationary trends.
// - If $y_t = nabla^d x_t$ (difference of order $d$), then $y_t$ follows an ARMA(p,q).

// == Handling trend and seasonality

// - Differencing: $nabla x_t = x_t - x_{t-1}$ (simple detrending).
// - Decomposition: $x_t = T_t + S_t + R_t$ (trend, seasonal, remainder) — methods: STL, additive/multiplicative decomposition.
// - SARIMA models for seasonality (seasonal p,d,q with period $s$).
// - Regression with exogenous variables (ARIMAX) to include covariates.

// == Alternative non-deep methods

// - Exponential smoothing (SES, Holt, Holt–Winters): strong practical performance for trend/seasonal series, simple to implement.
// - Transform-based methods (Box-Cox), decomposition, filtering (HP-filter).
// - Model selection: ACF/PACF, information criteria (AIC, BIC), time-series cross-validation.

// == Practical tips

// - Always visualize: trend/seasonality, ACF/PACF.
// - Check stationarity (ADF, KPSS); difference if necessary.
// - Prefer chronological validation (rolling / expanding windows) over random CV.
// - In modern pipelines: use forecasting as a training/evaluation signal for features/self-supervision.

// == References

// - Hyndman & Athanasopoulos — "Forecasting: principles and practice" (ARIMA, smoothing chapters).
// - Box & Jenkins — classical ARIMA theory.

