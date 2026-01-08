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

= Old-school models

== Additive decomposition

- Typical assumption: Time series can be decomposed into: Trend, Seasonality, and Residuals in an additive way:

  $
  x_t = T_t + S_t + R_t
  $

  #align(center)[
    #image("fig/stl.svg", width: 90%)
  ]

== ARIMA models

*Autoregressive AR(p)*

$
  x_t = c + sum_(i=1)^p phi_i x_(t-i) + epsilon_t
$
- Linear regression model based on a window of past values of size $p$.
- Captures temporal correlations via lags.

*Moving Average MA(q)*

$
  x_t = c + sum_(j=1)^q theta_j epsilon_(t-j) + epsilon_t
$
- Models the effect of past residuals on the current value.

#pagebreak()

*ARMA(p,q)*

$
  x_t = c + sum_(i=1)^p phi_i x_(t-i) + sum_(j=1)^q theta_j epsilon_(t-j) + epsilon_t
$
- Assumes stationarity.

*ARIMA(p,d,q)*

- Introduces differencing of order $d$ to handle non-stationary trends.
- If $y_t = nabla^d x_t$ (difference of order $d$), then $y_t$ follows an ARMA(p,q).

= Evaluation metrics and losses

== Metrics

*Mean Squared Error MSE*

$
  "MSE"(x_t, hat(x)_t) = 1/N sum_(t=1)^N ||x_t - hat(x)_t||_2^2
$

*Mean Absolute Error MAE*

$
  "MAE"(x_t, hat(x)_t) = 1/N sum_(t=1)^N ||x_t - hat(x)_t||_1
$

// *Mean Absolute Scaled Error MASE*

// $
//   "MASE"(x_t, hat(x)_t) = (1/N sum_(t=1)^N |x_t - hat(x)_t|)/(1/(N-1) sum_(t=2)^N |x_t - x_(t-1)|)
// $

*Symmetric Mean Absolute Percentage Error sMAPE*

$
  "sMAPE"(x_t, hat(x)_t) = 2/N sum_(t=1)^N (||x_t - hat(x)_t||_1) / (||x_t||_1 + ||hat(x)_t||_1)
$

== Differentiable losses

- MSE is the _de facto_ standard loss for forecasting
- Shape-based losses exist however, eg:

#align(center)[
  #image("fig/dtw_mse.svg", width: 100%)
]

== Dynamic Time Warping (DTW)

- Aligns two time series by warping the time axis
- Useful when similar patterns occur at different time locations
- Relies on solving the following optimization problem:

$
  "DTW"(X, hat(X)) = min_(pi in Pi) sum_(t=1)^T ||x_(pi_1(t)) - hat(x)_(pi_2(t))||_2^2
$

- Where $Pi$ is the set of all admissible alignments between the two series
- Can be solved via dynamic programming in $O(T^2)$ time

== soft-DTW

- DTW is not differentiable
- soft-DTW is a differentiable variant
- Replaces min by soft-min in the DP recursion:

$
  "softDTW"^gamma (X, hat(X)) = "softmin"^gamma_(pi in Pi) sum_(t=1)^T ||x_(pi_1(t)) - hat(x)_(pi_2(t))||_2^2
$

- Can be used as a loss in deep learning models
- Hyperparameter $gamma > 0$ controls the smoothness
  - $gamma -> 0$: soft-DTW approaches DTW
  - Large $gamma$: more smoothing (MSE-like behaviour)