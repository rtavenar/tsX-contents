#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": * // figure-placeholder

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning for Time Series - Continuous-Time Models],
  config-info(
    title: [Deep Learning for Time Series],
    subtitle: [Session 5: Continuous-Time Models],
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

== Motivation

#grid(columns: 2,
      gutter: 1em,
  image-with-caption(
    image("fig/abnormal-ecg.jpg", width: 100%),
    [#link("https://sunfox.in/blogs/understanding-abnormal-ecg-waves-with-spandan-ecg/?srsltid=AfmBOoptrPVqB3UW_A7RSvDVnELmCyTlxKBE-ywPvonzXkLTP8bK6YP4", [Source: sunfox.in blog])]
  ),
  image-with-caption(
    align(center)[
      #image("fig/bird-call_simplified.png", width:75%)
    ],
    [#link("https://www.nature.com/articles/s41598-023-33825-5", [Source: "Machine learning and statistical classification of birdsong link vocal acoustic features with phylogeny"])]
  )
)

---

- Time series are (almost always) discretization of continuous-time processes
- In real life, sensors fail
  - Missing data
  - Irregular sampling
- Basic neural architectures not tailored for such settings
  - Conv _vs_ Recurrent _vs_ Attention-based
  - Missing data imputation can help
*$=>$ Can we build neural architectures that operate in continuous time?*
  // - Bonus: if we have autodiff-ready functions of time, we can get $(partial f) / (partial t)$ for free (can be useful for some applications, eg. physics)

= Neural ODEs

== Ordinary Differential Equations (ODEs)

Assume the evolution of the forecast variable through time follows a system of the form:
$
cases(
  dot(x)(t) &= f(x(t), t),
  x(0) &= x_0
)
$

- Given $f$ and $x_0$, one can use an ODE solver to compute $x(t)$ for any $t$ by approximating:

$
  x(t) = x_0 + integral_0^t f(x(t), t) dif t
$

== A word on ODE solvers

#grid(columns: 2,
      gutter: 1em,
  [
    - Approximate the sequence of values:
    $
      [x(t+h), x(t+2h), dots , x(t)]
    $
    - Using a Taylor expansion to compute $x(tau + h)$ for $x(tau)$
    - First-order Taylor expansion gives the Euler method:
    $
      x(tau + h) approx x(tau) + h f(x(tau), tau)
    $
  ],
  image-with-caption(
    image("fig/ode_nfe.svg", width: 100%),
    [Solving a spiral ODE with a Euler scheme]
  )
)

---

- More advanced ODE solvers exist // (beyond the scope of this course)
  - Runge-Kutta methods (see below), adaptive step-size, ...

#align(center)[
  #image-with-caption(
    image("fig/ode_rk.svg", width: 70%),
    []
  )
]

== Neural ODEs

- ODEs for forecasting
  - Need to find the right $f$ for our data
  - Let $f$ be a neural network $f_theta$ and train it on our forecasting task
    - Need to compute gradients through the solver
    #align(center)[
      #image-with-caption(
        image("fig/node_gradients.svg", width: 50%),
        [Source: "Neural ODEs", NeurIPS'19]
      )
    ]
  - Euler scheme: akin to a ResNet with shared weights

== Latent NODEs

- Neural ODEs cannot operate in input space
  - Reconstruction from a single observation $x_0$ is a strong limitation
- Latent NODEs
  1. project in latent space to get higher-level information
  2. use RNN encoder to summarize information from the past
#align(center)[
  #image-with-caption(
    image("fig/node_latent.svg", width: 100%),
    [Source: "Neural ODEs", NeurIPS'19]
  )
]

---

#align(center)[
  #image-with-caption(
    image("fig/node_traj.svg", width: 50%),
    [Source: "Neural ODEs", NeurIPS'19]
  )
]

---

#align(center)[
  #image-with-caption(
    image("fig/node_latent.svg", width: 80%),
    []
  )
]
- RNN encoder above still assumes regular sampling

#align(center)[
  #image-with-caption(
    image("fig/node_latent_gru.svg", width: 80%),
    [Source: "Latent ODEs for Irregularly-Sampled Time Series", NeurIPS'20]
  )
]

== NODE-RNN

#grid(columns: 2,
      gutter: 1em,
  [
    - Standard RNN cell update:
    $
      h_(t) = "RNN"(h_(t-Delta t), x_t)
    $
    - NODE-RNN cell update:
    $
      tilde(h)_t &= "ODESolve"(h_(t-Delta t), Delta t) \
      h_(t) &= "RNN"(tilde(h)_t, x_t)
    $
  ],
  image-with-caption(
    image("fig/node_irregular_viz.svg", width: 100%),
    [Source: "Latent ODEs for Irregularly-Sampled Time Series", NeurIPS'20]
  )
)

= Implicit Neural Representations

== Basics

Implicit Neural Representations (INRs) model a signal as a continuous function of time:

$
  f_theta : RR &-> RR^d \
  t &|-> f_theta (t)
$

A common choice for $f_theta$ is an MLP.

--- 

- Representing time as a 1D feature is a weak representation
- In practice, an INR's first layer is often a positional encoding
  - Fourier features (random or learnt):
  $ h_1(t) =
  [ sin(omega_1 t), cos(omega_1 t), dots,
    sin(omega_K t), cos(omega_K t) ] $
  - Sine activated layer ($W$ is a learnable vector):
  $ h_1(t) = sin(omega_0 W t) $

== SIREN

- Use sine activation functions:
  $
    Phi(x) = sin(W dot x + b)
  $
- Acts as a learnable Fourier basis decomposition

#image-with-caption(
  align(center)[
    #image("fig/siren_bach.svg", width: 100%)
  ],
  [Source: "Implicit Neural Representations with Periodic Activation Functions", NeurIPS'20]
)

== Modulated INRs

- Such INRs can learn to extrapolate a single time series:
  $
    f_theta : RR &-> RR^d \
    t &|-> f_theta (t)
  $
- In practice, we have a dataset of time series, hence the formulation:
  $
    f_theta : RR times RR^p &-> RR^d \
    t, z &|-> f_theta (t, z)
  $
  where $z$ is a code summarizing the content of the time series
- *Modulated INRs*: $z$ modulates the behaviour of $f_theta$ (activations or weights)

== Feature-Wise Linear Modulation (FiLM)



#grid(columns: (65%, 1fr),
      gutter: 1em,
  [
    1. The time series is encoded as $z$
    2. A modulation network (shallow MLP) outputs modulation parameters $gamma(z)$ and $beta(z)$ for each INR layer
    3. These parameters are used to modulate the INR *activations*
    $
      h^"modulated" = gamma(z) dot.o h^"INR" + beta(z)
    $
  ],
  image-with-caption(
    image("fig/film.svg", width: 100%),
    [Source: "FiLM [...]", AAAI'18]
  )
)

== Hypernetworks

- Modulation at the INR *parameter* level:
  1. A hypernetwork learns per-parameter modulations $psi(z)$
  2. The INR (hyponetwork) is now:
  $
    f_(theta, psi) : RR times RR^p &-> RR^d \
    t, z &|-> f_(theta + psi(z)) (t)
  $

== Typical TS forecasting INRs

*HyperTime*
- Generates an encoding $z$ per timestamp
- Requires global pooling along the time axis (or fixed number of observations per series)

#image-with-caption(
  align(center)[
    #image("fig/hypertime.svg", width: 100%)
  ],
  [Source: "HyperTime: Implicit Neural Representation for Time Series", NeurIPS Workshop, 2022]
)

---

*TimeFlow*
- Code $z$ is optimized through few-step gradient descent \ (not output by an encoder) \
  $->$ no constraint on the input time grid, no need for pooling

#image-with-caption(
  align(center)[
    #image("fig/timeflow.svg", width: 100%)
  ],
  [Source: "Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations", TMLR'24]
)
