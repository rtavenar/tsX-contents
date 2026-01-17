#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": * // figure-placeholder

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning for Time Series - Basics],
  config-info(
    title: [Deep Learning for Time Series],
    subtitle: [Session 3: Attention-based architectures],
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

#title-slide()

= Intro

== Self-attention vs RNNs vs ConvNets

#align(center)[
  #include "cetz/inductive_biases.typ"
]

== Illustration of self-attention

#figure-placeholder(100%, 5em)

== Basic transformer architecture

#figure-placeholder(100%, 5em)

== Encoder-only vs Decoder-only for forecasting

#align(center)[
  #image-with-caption(
    image("fig/encoder_decoder_forecasting.svg", width: 100%),
    [Source: "Towards Neural Scaling Laws for Time Series Foundation Models" (ICLR'25)]
  )
]

- Encoder-only (bi-directional attention) is more versatile (can be applied to different tasks)
- Decoder-only is more suitable for forecasting

== Failures of transformers in forecasting

#align(center)[
  #image-with-caption(
    image("fig/are_transformers_effective.svg", width: 60%),
    []
  )
]

- Comparison between simple linear predictor and SOTA transformers (at the time)

#align(center)[
  #image-with-caption(
    image("fig/linear_model.svg", width: 40%),
    [Source: "Are Transformers Effective for Time Series Forecasting?" (AAAI'23)]
  )
]

#uncover("2-")[
  #place(
    top + left,
    dx: 15%,
    dy: 30%,
    box(fill: rgb(131,109,169), inset: 1em, radius: 0.5em, width: 70%)[
      #align(center)[
        #text(color.rgb(255, 255, 255))[
          "LTSF-Linear surprisingly outperforms existing sophisticated Transformer-based LTSF models in all cases, and often by a large margin"
        ]
      ]
    ]
  )
]

== PatchTST: a transformer that works well

- Previous methods:
  1. each patch = 1 time point or some extracted information
  2. all variates are considered together
- PatchTST:
  1. patch = subseries
  2. channel-independence \
     (bonus: much easier for transfer learning)
  3. proper preprocessing (RevIN)

#pagebreak()

*Channel independence*

#align(center)[
  #image-with-caption(
    image("fig/patchtst_channel.svg", width: 100%),
    [Source: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR'23)]
  )
]

#pagebreak()

*Transformer backbone*

#align(center)[
  #image-with-caption(
    image("fig/patchtst_backbone.svg", width: 50%),
    [Source: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR'23)]
  )
]

#pagebreak()

*Preprocessing (RevIN)*

$
  hat(x)_(d,t)^((i)) = gamma_d dot (x_(d,t)^((i)) - hat(mu)_(d)^((i))) / hat(sigma)_d^((i)) + beta_d " and reverse before forecasting"
$

#align(center)[
  #image-with-caption(
    image("fig/revin.svg", width: 80%),
    [Source: "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift" (ICLR'22)]
  )
]

== More insights on transformer failures

// - From paper by Odonnat et al. on SAMformer
- Transformers have a sharp loss landscape:

#align(center)[
  #image-with-caption(
    image("fig/sam.svg", width: 100%),
    [Source: "When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations" (ICLR'22)]
  )
]

- Train with SAM (Sharpness-Aware Minimization) to smooth the landscape:

$
  cal(L)_"SAM" = max_(||epsilon||_2 <= rho) cal(L)(w + epsilon)
$

#pagebreak()

- SAMformer: a forecasting Transformer trained with SAM

#align(center)[
  #image-with-caption(
    grid(
      columns: 2,
      gutter: 1em,
      image("fig/samformer.svg", height: 10em),
      image("fig/samformer_arch.svg", height: 8em),
    ),
    [Source: "SAMformer: [...]" (ICML'24)]
  )
]

== Conclusion

- Transformers can work well for time series forecasting
- But need proper design choices:
  1. patch-based input
  2. channel-independence
  3. proper preprocessing (RevIN)
  4. smooth optimization (SAM)
