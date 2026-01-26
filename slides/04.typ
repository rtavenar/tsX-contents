#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": * // figure-placeholder

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning for Time Series - Basics],
  config-info(
    title: [Deep Learning for Time Series],
    subtitle: [Session 4: State Space Models],
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

- RNNs: sequential, hard to scale
- CNNs: local receptive field
- Transformers:
  - expressive
  - but $O(L^2)$ in sequence length

#figure-placeholder(100%, 5em)

---

== A question

Can we get:
- long-range dependencies
- linear (or near-linear) complexity
- strong inductive bias for sequences?

→ *State Space Models*

---

= State Space Models

== Classical SSM formulation

Latent state $x_t in RR^N$ evolves over time:

$
  x_(t+1) &= A x_t + B u_t \
  y_t &= C x_t
$

- $u_t$: input
- $y_t$: output

#figure-placeholder(100%, 5em)

---

== Intuition

- $x_t$: compressed memory of the past
- Linear dynamics over time
- Used for decades in control & signal processing

#figure-placeholder(100%, 5em)

---

== From control to deep learning

Key idea:
- Learn $(A, B, C)$
- Make SSMs differentiable
- Scale them to large datasets

SSMs become *sequence layers*

---

= SSMs as sequence layers

== Convolutional view

A linear SSM defines a *convolution*:

$
  y = K * u
$

where:
- $K$ is implicitly defined by $(A, B, C)$
- Can be computed efficiently

#figure-placeholder(100%, 5em)

---

== Why this matters

- Parallel over time (like Transformers)
- Linear complexity in sequence length
- Strong inductive bias for long memory

---

= Modern SSMs

== The SSM renaissance

Recent models:
- S4
- DSS
- Mamba
- Hyena (related ideas)

All build on the same core principle.

#figure-placeholder(100%, 5em)

---

== Key design choices

- Structure of matrix $A$
- Parameterization for stability
- Efficient kernel computation
- Gating & nonlinearities

(Details at the board.)

---

== Example: diagonal SSMs

Assume $A$ is diagonal (or diagonalizable):

- Dynamics decouple across dimensions
- Fast computation
- Stable training

This is the core idea behind *S4*

---

= Mamba & selective SSMs

== Limit of pure linearity

Linear SSMs:
- great memory
- limited expressivity

Solution:
→ *input-dependent dynamics*

---

== Selective State Space Models

Mamba introduces:
- input-dependent $B_t$, $C_t$
- gating mechanisms
- still linear-time

#figure-placeholder(100%, 5em)

---

== Why Mamba works so well

- Acts like a content-aware filter
- Keeps SSM efficiency
- Competitive with Transformers on long sequences

---

= Takeaways

== When should you use SSMs?

- Very long sequences
- Memory matters more than attention
- Efficiency is critical

---

== Mental model

- Transformers: dynamic attention over tokens
- SSMs: learned dynamical systems over time

Both are sequence models — different biases.

#figure-placeholder(100%, 5em)

---

== Final message

SSMs are not new.  
But *now* we know how to scale them.

🚆 Get on the SSM train.

