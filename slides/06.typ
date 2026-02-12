#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": * // figure-placeholder

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning for Time Series - State Space Models],
  config-info(
    title: [Deep Learning for Time Series],
    subtitle: [Session 6: State Space Models],
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

#set text(font: "Helvetica Neue", weight: "light")
#show link: underline

#title-slide()

= Motivation

== Caveats of previously introduced models

#align(center)[
  #include "cetz/inductive_biases_nocausal.typ"
]

- RNNs: sequential, limits parallelism
- CNNs: local receptive field
- Attention-based models: $O(T^2)$ complexity

== What State Space Models (SSMs) can offer

- Continuous-time dynamics
- Long-range dependencies
- Linear (or near-linear) complexity

#align(center)[
  #image-with-caption(
    image("fig/ssm_3views.png", width: 100%),
    [Source: "Structured State Spaces: Combining Continuous-Time, Recurrent, and Convolutional Models", a blogpost by by Albert Gu et al. (2022)]
  )
]

= State Space Models

== Classical SSM formulation

Latent state $h_t in RR^d$ evolves over time following:

$
  cases(
    dot(h) (t) &= A h (t) + B x (t),
    o (t) &= C h (t) + cancel(D x(t))
  )
$

- $x (t)$: input
- $o (t)$: output
- $h (t)$: latent state

In practice, we often set $D=0$ for simplicity (can be later recovered through residual connection in Deep SSM anyway).

== Step 1: Time discretization (S4)

#align(center)[
  #image-with-caption(
    image("fig/ssm_discretization.png", width: 70%),
    [Source: "Structured State Spaces: Combining Continuous-Time, Recurrent, and Convolutional Models", a blogpost by by Albert Gu et al. (2022)]
  )
]

---

$
  dot(h) (t) &= A h (t) + B x (t)
$

Let us discretize time with step $Delta$ between $t$ and $t+1$.
1. Textbook ODE solving gives the following form for $h(t)$:

$
  h (t) &= e^(A Delta) h (t-Delta) + integral_0^Delta e^(A s) B x (t - s) dif s
$

#pause

2. Assume $x(t)$ is constant between $t - Delta$ and $t$: // (i.e., zero-order hold):

$
  h (t) &= e^(A Delta) h (t-Delta) + (integral_0^Delta e^(A s) dif s) B x (t)
$

#pause

3. Get the discrete update rule:

$
  h_(t) &= overline(A) h_(t-1) + overline(B) x_(t)
$

---

Now we have the discrete system:

$
  cases(
    h_(t) &= overline(A) h_(t-1) + overline(B) x_(t),
    o_(t) &= overline(C) h_(t)
  )
$

with:
- $overline(A) = e^(A Delta)$
- $overline(B) = (integral_0^Delta e^(A s) dif s) B$
- $overline(C) = C$

*Main idea behind S4*: Parametrize $A$, $B$, $C$ such that the resulting discrete system has good properties (efficient computation, stable training, long memory).

// TODO here: on the right of the figure, add an illustration of the discrete system (boxes for $h_t$, arrows for the updates, etc.) to make it more concrete.

---

*Efficient computation: The convolutional trick*

Let us assume $h_0 = 0$, then we get:

$
  h_t &= sum_(k=1)^(t) overline(A)^(t-k) overline(B) x_k
$

#v(1em)

#align(center)[
  #scale(x: 160%, y: 160%,
    include "cetz/conv1d_ssm.typ",
  )
]

#v(3em)

Efficient implementation: compute the convolution using FFT

--- 

*Long-term memory: The parametrization trick*

- Risk of vanishing for long-range terms (i.e., $overline(A)^k overline(B)$ could vanish for large $k$)
- S4's solution: smart parametrization of $A$
  - Typical choice: $A = V Lambda V^(-1)$ with eigenvalues of the form 
    $
      lambda_k (A) = -underbrace(alpha_k, > 0) + i omega_k
    $
  - $overline(A) = e^(A Delta) = V e^(Lambda Delta) V^(-1)$ with $lambda_k (overline(A)) = e^(-alpha_k Delta) e^(i omega_k Delta)$
  - $overline(A)^k = V (e^(Lambda Delta))^k V^(-1)$ 
  - If $alpha_k$ is small, we get long-range dependencies without vanishing (since $e^(-alpha_k Delta)$ is close to 1)

---

*Recovering continuous-time dynamics*

- Easy since S4 parametrizes $A$ directly 
- Just need to compute
  $
    h (t) &= e^(A Delta) h (t-Delta) + (integral_0^Delta e^(A s) dif s) B x (t)
  $
  on a sufficiently fine-grained grid (remember the constant $x(t)$ hypothesis)
- We have:
  $
    integral_0^Delta e^(A s) dif s = A^(-1) (e^(A Delta) - I)
  $
  since $A$ is invertible, we're good to go!

== Step 2: Input-dependent dynamics (Mamba)

Pure linearity in SSMs:
- ✓ Great long-range memory
- ✗ Limited expressivity

*Solution: Make dynamics input-dependent*

$
  cases(
    h_(t) &= e^(-Delta_t A) h_(t-1) + B_t x^prime_(t),
    o_(t) &= C_t h_(t)
  )
$
with $Delta_t$, $B_t$, $C_t$, $x_t^prime$ linear in the input $x_t$

Now the state transition adapts to the input as in attention-based models

---

== Mamba: Efficiency without convs

The challenge: input-dependent parameters break the convolutional structure.

Mamba's design choices:

1. Scan-based computation
  - Process sequentially (like RNN)
  - Still efficient thanks to low-level optimizations (e.g., GPU kernels) using the linearity of the state update
2. Gating mechanisms
  - Controls what information is retained in the state
  - Similar to LSTM/GRU gates

$=>$ $O(T)$ complexity


== Mamba4Cast: a foundation model for (univariate) TS forecasting

#image-with-caption(
  image("fig/mamba4cast.svg", width: 100%),
  [
    Source: "Mamba4Cast: Efficient Zero-Shot Time Series
Forecasting with State Space Models", NeurIPS'24 Workshop
  ]
)
