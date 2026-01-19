#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": * // figure-placeholder

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning for Time Series - Basics],
  config-info(
    title: [Deep Learning for Time Series],
    subtitle: [Session 2: ConvNets and Recurrent architectures],
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

= Convolutional architectures

== ConvNets 101

- Basic time series processing: 1d convolutions (over time)
- Limited receptive field: co-localization matters

  #align(center)[
    #scale(120%)[
      #include "cetz/conv1d.typ"
    ]
  ]

== Causal convolutions

- Forecasting tasks: cannot access the future
- Causal convolution: convolve on past information alone (asymmetric window)

  #align(center)[
    #scale(120%)[
      #include "cetz/conv1d_causal.typ"
    ]
  ]

== Temporal Convolution Network (TCN)

- Main idea: cascade dilated causal convolutions \
  $=>$ Larger receptive field

#align(center)[
  #image-with-caption(
    image("fig/tcn_dilatedconvs.svg", width: 70%),
    [Source: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", Bai et al., arXiV 2018]
  )
]

== Temporal Convolution Network

- Additional improvements:
  - Residual connections \
    $=>$ Multi-resolution analysis
  - Normalization+Dropout layers

#align(center)[
  #image-with-caption(
    image("fig/tcn_block.svg", width: 40%),
    [Source: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", Bai et al., arXiV 2018]
  )
]

= Recurrent architectures

== Recurrent Neural Networks (RNNs)

- Very flexible model (any length, let the model learn its memory needs, ...)

#image-with-caption(
  image("fig/rnn_unroll.svg"),
  [Source: #link("https://colah.github.io/posts/2015-08-Understanding-LSTMs/")[Christopher Olah's blog]]
)

== "Vanilla" RNN

- Hidden state is computed as:
$
  h_t = phi(#grad-disk())
$

#align(center)[
  #image-with-caption(
    image("fig/rnn_cell.svg", width: 80%),
    [Source: #link("https://colah.github.io/posts/2015-08-Understanding-LSTMs/")[Christopher Olah's blog]]
  )
]

#place(
  right + bottom,
  dx: -14.6em,
  dy: -7.2em
)[
  #grad-disk()
]

#place(
  right + bottom,
  dx: 2em
)[
  #grad-disk(from: xi, to: xi) $x_t$
  #grad-disk(from: yj, to: yj) $h_(t-1)$
  #grad-disk() Linear combination of $x_t$ and $h_(t-1)$
]

== "Vanilla" RNN properties

- Very flexible model (any length, let the model learn its memory needs, ...)
- Difficult to learn in practice
  - Slow (lack of parallelism)
  - Vanishing gradients (hard to learn long-term dependencies) or exploding gradients (if $phi$ is unbounded)
#image-with-caption(
  image("fig/rnn_unroll.svg"),
  [Source: #link("https://colah.github.io/posts/2015-08-Understanding-LSTMs/")[Christopher Olah's blog]]
)

== Gated Recurrent Unit (GRU)

- At each time step, keep only part of the information
  - Through *gating mechanism*

#grid(columns: 2,
      gutter: 1em,
  image-with-caption(
    image("fig/gru.svg", width: 80%),
    [Source: #link("https://colah.github.io/posts/2015-08-Understanding-LSTMs/")[Christopher Olah's blog]]
  ),
  $
    z_t &= sigma(#grad-disk(angle: 45deg)) " (update gate)" \
    r_t &= sigma(#grad-disk(angle: -45deg)) " (reset gate)" \
    tilde(h)_t &= phi(W dot x_t + R dot [r_t dot.o h_(t-1)]) \
    h_t &= (1 - z_t) dot.o h_(t-1) + z_t dot.o tilde(h)_t
  $
)

#place(
  right + bottom,
  dx: 2em
)[
  #grad-disk(from: xi, to: xi) $x_t$
  #grad-disk(from: yj, to: yj) $h_(t-1)$
  #grad-disk() Linear combination of $x_t$ and $h_(t-1)$
]

== Long Short Term Memory (LSTM)

- Similar ideas as in GRUs, but:
  - an additional _cell state_ $C_t$
    #align(center)[
      #image-with-caption(
        image("fig/lstm_1.svg", width: 80%),
        [Source: #link("https://colah.github.io/posts/2015-08-Understanding-LSTMs/")[Christopher Olah's blog]])
    ]
  - input and forget gates are made independent \
    (in place of $z_t$ in GRU)

== Long Short Term Memory (LSTM)

- *Forget gate*: $f_t = sigma(#grad-disk())$

#align(center)[
  #image-with-caption(
    image("fig/lstm_2.svg", width: 60%),
    [Source: #link("https://colah.github.io/posts/2015-08-Understanding-LSTMs/")[Christopher Olah's blog]])
]

#place(
  right + bottom,
  dx: 2em
)[
  #grad-disk(from: xi, to: xi) $x_t$
  #grad-disk(from: yj, to: yj) $h_(t-1)$
  #grad-disk() Linear combination of $x_t$ and $h_(t-1)$
]

== Long Short Term Memory (LSTM)

- *Input gate*: $i_t = sigma(#grad-disk(angle: 45deg))$
- *Suggested $C_t$ update*: $tilde(C_t) = phi(#grad-disk(angle: -45deg))$

#align(center)[
  #image-with-caption(
    image("fig/lstm_3.svg", width: 60%),
    [Source: #link("https://colah.github.io/posts/2015-08-Understanding-LSTMs/")[Christopher Olah's blog]])
]

#place(
  right + bottom,
  dx: 2em
)[
  #grad-disk(from: xi, to: xi) $x_t$
  #grad-disk(from: yj, to: yj) $h_(t-1)$
  #grad-disk() Linear combination of $x_t$ and $h_(t-1)$
]

== Long Short Term Memory (LSTM)

- *$C_t$ update rule*: $C_t = f_t dot.o C_(t-1) + i_t dot.o tilde(C_t)$

#align(center)[
  #image-with-caption(
    image("fig/lstm_4.svg", width: 60%),
    [Source: #link("https://colah.github.io/posts/2015-08-Understanding-LSTMs/")[Christopher Olah's blog]])
]

== Long Short Term Memory (LSTM)

- *Output gate*: $o_t = sigma(#grad-disk(angle: 90deg))$
- *Hidden state update rule*: $h_t = o_t dot.o phi(C_t)$

#align(center)[
  #image-with-caption(
    image("fig/lstm_5.svg", width: 60%),
    [Source: #link("https://colah.github.io/posts/2015-08-Understanding-LSTMs/")[Christopher Olah's blog]])
]

#place(
  right + bottom,
  dx: 2em
)[
  #grad-disk(from: xi, to: xi) $x_t$
  #grad-disk(from: yj, to: yj) $h_(t-1)$
  #grad-disk() Linear combination of $x_t$ and $h_(t-1)$
]



== xLSTM

- A "modern" LSTM variant
  - Made of sLTSM and mLSTM layers
  - Embedded in blocks with normalization layers, residual connections, _à la_ Transformer

#align(center)[
  #image-with-caption(
    image("fig/xlstm.svg", width: 80%),
    [Source: "xLSTM: Extended Long Short-Term Memory" by Beck et al., NeurIPS 2024]
  )
]

== xLSTM

- What's "new"?
  - In both sLSTM and mLSTM layers:
    - Exponential activation (to face vanishing gradients)
  - In sLTSM only:
    - Multi-head
  - In mLSTM only:
    - Novel memory store
    - Drop recurrence for gate computations: better parallelism


== xLSTM: focus on sLSTM layers

- Exponential activation for input and forget gates:
  $ 
    i_t &= exp(#grad-disk(angle: 45deg)) \
    f_t &= max(exp(#grad-disk()), sigma(#grad-disk()))
  $
  $=>$ Need normalization:
  $ 
    n_t &= f_t dot.o n_(t-1) + i_t \
    h_t &= o_t dot.o C_t div.o n_t
  $
- Multi-head: keep separate linear combinations per head

#place(
  right + bottom,
  dx: 2em
)[
  #grad-disk(from: xi, to: xi) $x_t$
  #grad-disk(from: yj, to: yj) $h_(t-1)$
  #grad-disk() Linear combination of $x_t$ and $h_(t-1)$
]

== xLSTM: focus on mLSTM layers

- Exponential activation as in sLSTM
- Memory store \
  $
    C_t &= f_t dot.o C_(t-1) + i_t dot.o v_t k_t^top \
    tilde(h_t) &= C_t q_t #h(9em) " (up to normalization)" #h(-4em) 
  $
  - Simplified case (no gate): similar to QKV in self-attention
- Drop recurrence for gate computations: better parallelism
  $
    i_t &= exp(#grad-disk(from: xi, to: xi)) \
    f_t &= max(exp(#grad-disk(from: xi, to: xi)), sigma(#grad-disk(from: xi, to: xi))) \
    o_t &= sigma(#grad-disk(from: xi, to: xi))
  $

#place(
  right + bottom,
  dx: 2em
)[
  #grad-disk(from: xi, to: xi) $x_t$
  #grad-disk(from: yj, to: yj) $h_(t-1)$
  #grad-disk() Linear combination of $x_t$ and $h_(t-1)$
]

== xLSTM: Building blocks

#columns(2, gutter: 8pt)[
  #image-with-caption(
    image("fig/block_sLSTM.svg", height: 80%),
    [An sLSTM block],
    caption-align: center
  )

  #colbreak()

  #image-with-caption(
    image("fig/block_mLSTM.svg", height: 80%),
    [An mLSTM block],
    caption-align: center
  )
]
