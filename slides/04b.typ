#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": * // figure-placeholder

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning for Time Series - Time Series Foundation Models],
  config-info(
    title: [Deep Learning for Time Series],
    subtitle: [Session 4b: Time Series Foundation Models],
    author: [Romain Tavenard],
    institution: [NB: Some figures in this slide deck are borrowed from Ievgen Redko's (great) course on time series forecasting and classification]
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



== Traditional Machine Learning


#align(center)[
#image-with-caption(
    image("fig/ml-pipeline.svg", width: 100%),
    []
  )
]

- Traditional ML pipeline
  1. Collect training data
  2. Train a new model
  3. Deploy it

---

#align(center)[
#image-with-caption(
    image("fig/ml-pipeline.svg", width: 100%),
    []
  )
]

- Traditional ML pipeline
  1. Collect training data $<-$ Issue 1: requires large training set
  2. Train a new model $<-$ Issue 2: a new model for each task
  3. Deploy it

== Success of pre-training

- In Computer Vision: ImageNet pre-training is everywhere
- In text: multi-task LLMs
- In time series
  - Is there an ImageNet?
  - Do we have competitive task-adaptive models?

== Time Series Foundation Models (TSFMs)

#align(center)[
#image-with-caption(
    image("fig/tsfm_overview.svg", width: 100%),
    []
  )
]

= Step 1: Pre-training

== Pre-training datasets

- #link("https://huggingface.co/datasets/Salesforce/lotsa_data", [LOTSA (MOIRAI)])
  - 27.7B observations
  - 4M time series
  - LOTSAMini exists: < 1% of LOTSA
- #link("https://huggingface.co/datasets/AutonLab/Timeseries-PILE", [TimeSeries-PILE (MOMENT)])
  - 1.1B observations
  - 13M time series
- #link("https://huggingface.co/datasets/Maple728/Time-300B", [Time300B])
  - 300B observations
  - 48M time series
- TimePFN's synthetic data generators
  - used 300-600M observations for TimePFN pre-training

== Pre-training strategies

- Pre-training across tasks
  #align(center)[
  #image-with-caption(
      image("fig/moment_tspile.svg", width: 60%),
      [Source: "MOMENT: A Family of Open Time-series Foundation Models", ICML'24]
    )
  ]
- Semi-Supervised Learning
  - eg. MOMENT (masking), TS2Vec (contrastive)
- Use pre-trained LLM
  - eg. GPT4TS

== TS2Vec

- Self-supervised contrastive learning for time series
  - Contrast positives: same series, different segments/scales
  - Contrast negatives: different time series
- Learns multi-scale (hierarchical) representations
- Temporal encoder (CNN-based), no decoder
- Outputs fixed-length embeddings for downstream tasks

#align(center)[
#image-with-caption(
    image("fig/ts2vec.svg", width: 60%),
    [Source: "TS2Vec: Towards Universal Representation of Time Series", AAAI'22]
  )
]

---

- First strong task-agnostic TS representation method
- Validated pretrain once $->$ reuse everywhere paradigm
- Shifted SSL from reconstruction to semantic contrastive learning
- Strong zero-shot / linear-probe baseline

== GPT4TS

General idea: make use of LLM backbone
1. Normalization + Patching (_à la_ PatchTST)
2. Projection to LLM embedding dimension
3. Fine-tuning of positional embeddings + layer normalization, keeping others frozen

#align(center)[
#image-with-caption(
    image("fig/gpt4ts.svg", width: 60%),
    [Source: "One Fits All: Power General Time Series Analysis by Pretrained LM", NeurIPS'23]
  )
]


= Step 2: Adapting

== TabPFN-TS: In-Context learning

- TabPFN is pre-trained on tabular data alone
- Builds basic TS features (calendar, seasonal, ...)
- No TS prior at all!

#align(center)[
#image-with-caption(
    image("fig/tabpfn.svg", width: 100%),
    [Source: "From Tables to Time:
Extending TabPFN-v2 to Time Series Forecasting", ArXiV'26]
  )
]

== MOMENT

Encoder-only model: can be used for any downstream task (classification, forecasting, imputation etc)

#align(center)[
#image-with-caption(
    image("fig/moment_archi.svg", width: 80%),
    [Source: "MOMENT: A Family of Open Time-series Foundation Models", ICML'24]
  )
]

== MANTIS: a FM for TSC

- Contrastive pre-training
- Scalar embedding module for preserving basic statistics
- Strong zero-shot and fine-tuning performance for TSC

#align(center)[
#image-with-caption(
    image("fig/mantis.svg", width: 90%),
    [Source: "Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification", ICML workshops'25]
  )
]

// == xLSTM-Mixer

// - RevIN + Baseline channel-wise linear forecast
// - sLSTM blocks
// - Mixer principle at all stages

// #align(center)[
// #image-with-caption(
//     image("fig/xlstm-mixer.svg", width: 90%),
//     [Source: "xLSTM-Mixer: Multivariate Time Series Forecasting
// by Mixing via Scalar Memories", NeurIPS'25]
//   )
// ]

== TiRex

- Masking-based pre-training
- Simple xLSTM architecture (only sLSTM blocks)

#align(center)[
#image-with-caption(
    image("fig/tirex.svg", width: 90%),
    [Source: "TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning", NeurIPS'25]
  )
]

= Step 3: Evaluating

== GIFTEval — Benchmarking Time-Series Foundation Models

- Unified evaluation benchmark for TSFMs
- Focuses on forecasting
- Diverse datasets across domains and time scales
  - short / long term
  - univariate / multivariate
- https://huggingface.co/spaces/Salesforce/GIFT-Eval


