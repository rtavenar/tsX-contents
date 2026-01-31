#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": * // figure-placeholder

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning for Time Series - Time Series Classification],
  config-info(
    title: [Deep Learning for Time Series],
    subtitle: [Session 4: Time Series Classification],
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

= Basics of Time Series Classification (TSC)

== TSC examples

#grid(columns: 2,
      gutter: 1em,
  image-with-caption(
    image("fig/abnormal-ecg.jpg", width: 100%),
    [#link("https://sunfox.in/blogs/understanding-abnormal-ecg-waves-with-spandan-ecg/?srsltid=AfmBOoptrPVqB3UW_A7RSvDVnELmCyTlxKBE-ywPvonzXkLTP8bK6YP4", [Source: sunfox.in blog])]
  ),
  image-with-caption(
    image("fig/bird-call.png", width: 100%),
    [#link("https://www.nature.com/articles/s41598-023-33825-5", [Source: "Machine learning and statistical classification of birdsong link vocal acoustic features with phylogeny"])]
  )
)

// == Problem setup

// - Time series classification: given a time series, predict its class
// - Univariate vs. multivariate

// ![Image 9-1](PLACEHOLDER)

== Specific challenges of Multivariate Time Series Classification

- How to combine channels?
  - Early fusion
  - Late fusion
  - Channel-wise models

- Correlations between variables matter
  - Ignoring them hurts performance

== Dataset-specific challenges

- Typical TSC datasets are *small*
  - Often a few hundreds of training samples
  - High risk of overfitting for deep models

- Strong heterogeneity across datasets
  - Length: short vs. long series
  - Noise levels
  - Intra-class variability

// - Class imbalance is common
//   - Accuracy can be misleading
//   - Balanced accuracy sometimes preferred

// ---

== Benchmarking practices

- UCR/UEA benchmark is widely used

- Fixed train / test splits
  - No cross-validation in standard benchmarks
  - Encourages benchmark-specific tuning

- Evaluation often relies on:
  - Accuracy
  - Average rank across (very diverse) datasets

$=>$ Risk of overfitting to benchmarks rather than solving real-world problems

= Overview of the State-of-the-art

== Bird's eye view

1. Before 2016:
  - Feature-based methods
  - Distance-based methods
  - Ensembles (COTE series)
2. 2016-2020
  - DNNs used as is (Resnets, MLPs, CNNs, FCNs, InceptionTime)
  - Random convolutions: Rocket (2020), MiniRocket (2021), MultiRocket (2022)
  - Transformers  //: as last time but with a classification head
  - HIVE-COTEv1 and v2: ensembling of more methods + hierarchical vote (not covered)
3. 2022- : Foundation models (covered in the next session)

== Remarks

1. Contrary to forecasting, DNNs are yet to beat other baselines
2. Transformer-based methods are mainly derivations of forecasting models
3. Ensemble methods are very competitive in benchmarks

= Historical baselines

== Distance-based baselines

- Key similarity measures (task-specific):
  - Euclidean distance
  - (variants of) Dynamic Time Warping
  - Longest Common Subsequence
  - _etc._

// PLACEHOLDER HERE: a figure with Euclidean, DTW, and subsequence matching

== Feature-based baselines

- Extract numerous features and plug a simple classifier
  - Often a strong baseline in benchmark evaluations!

#grid(columns: 3,
      gutter: 1em,
  image-with-caption(
    image("fig/hctsa.png", width: 100%),
    []
  ),
  image-with-caption(
    image("fig/catch22.png", width: 80%),
    []
  ),
  image-with-caption(
    image("fig/tsfresh.png", width: 100%),
    []
  )
)

- Example features:
  - `sum_of_reoccurring_values` - sum of all values present in the time series more than once
  - `longest_strike_above_mean` - length of the longest consecutive subsequence that is bigger than the mean
  

== Shapelets

- Shapelet = a subsequence of consecutive observations from a time series
  - Can be chosen or learned
  - Goal: Choose/learn a pool of K shapelets that are discriminative for a given task

#grid(columns: 2,
      gutter: 1em,
  image-with-caption(
    image("fig/shapelets_0.svg", width: 100%),
    []
  ),
  image-with-caption(
    image("fig/shapelets_1.svg", width: 100%),
    [Illustrations from #link("https://tslearn.readthedocs.io/en/latest/user_guide/shapelets.html", [tslearn docs])]
  )
)

== From classic to learned shapelets

- Original shapelets:
  - Exhaustive or heuristic search
  - Expensive but interpretable

- Learned shapelets:
  - Learning Shapelets (LS)
  - Shapelet layers in neural networks
  - Joint optimization with classifier

- Advantages:
  - Interpretability
  - Competitive performance on small datasets

== A word on Shapelet interpretability

- Shapelets are often selected for their interpretability
- What about learned shapelets?

#align(center)[
#image-with-caption(
    image("fig/shapelets_interpretability.svg", width: 80%),
    [Source: "Localized Random Shapelets", AALTD'19]
  )
]

== Ensembles

- Collective of Transformation-based Ensembles (COTE)
  - if there is no prior knowledge, ensemble different representations

1. Flat-COTE (2016): 35 classifiers over four data representations
  - shapelets, DTWs, _etc_
2. Hive-COTE-alpha,v1,v2 (2018, 2020, 2022):
  - more representations (forests, spectral) + hierarchical voting procedure
  
HC2 is one of the best methods on open public benchmarks but very slow

= From traditional models to deep learning

== Take 1: just try classic vision models on TS

- Many standard Conv-based architectures can be adapted for TSC (eg. InceptionTime: a Resnet with inception module)
    - a stack of convolutions of different sizes
    $->$ multi-resolution analysis

#align(center)[
#image-with-caption(
    image("fig/inception-time.svg", width: 60%),
    [Source: "InceptionTime: Finding AlexNet for Time Series
Classification", DMKD'20]
  )
]

== Take 2: simple models can be strong baselines

- ROCKET: use random 1D convolutions as feature extractors
  - Use maxpooling and PPV (proportion of positive values) as aggregators
  - Apply ridge regressor to the obtained embedding

#align(center)[
#image-with-caption(
    image("fig/rocket.png", width: 100%),
    [Source: #link("https://www.aeon-toolkit.org/en/stable/examples/classification/convolution_based.html", [aeon docs])]
  )
]

--- 

- ROCKET extensions:
  - hard-coded convolutions (MiniRocket)
  - more aggregators (MultiRocket)
  - random convolutions + dictionary learning (MR-HYDRA)

$->$ On par with HC2 but much faster


== TimesNet: a CNN with inception module and 2D kernels

- Assumption: the signal is periodic
- Motivation: we want to capture intra-period AND inter-period variations

#align(center)[
#image-with-caption(
    image("fig/timesnet.png", width: 100%),
    [Source: "TimesNet: [...]", ICLR'23]
  )
]


---

#align(center)[
#image-with-caption(
    image("fig/timesnet_perfs.svg", width: 90%),
    [Source: "TimesNet: [...]", ICLR'23]
  )
]

// NB: note all other transformers seen last time in the forecasting-orientated lecture

== Accuracy vs. Efficiency Trade-offs: Computational considerations

- COTE / HC2:
  - Very strong accuracy (favored by the diversity of the benchmarks)
  - Extremely expensive computationally

- ROCKET-based methods:
  - Fast training and inference
  - Excellent accuracy-efficiency trade-off

- Deep models:
  - GPU-friendly, data-hungry



// ## Page 32

// Foundation models

// ## Page 33

// Time Series Foundation Model (TSFM)
// Step 1: Pre-training Step 2: Fine-tuning to New Task
// Option 1: Option 2:
// +
// Head
// Random Forest
// prediction
// prediction

// ![Image 33-1](PLACEHOLDER)

// ![Image 33-2](PLACEHOLDER)

// ![Image 33-3](PLACEHOLDER)

// ![Image 33-4](PLACEHOLDER)

// ![Image 33-5](PLACEHOLDER)

// ![Image 33-6](PLACEHOLDER)

// ![Image 33-7](PLACEHOLDER)

// ![Image 33-8](PLACEHOLDER)

// ![Image 33-9](PLACEHOLDER)

// ![Image 33-10](PLACEHOLDER)

// ![Image 33-11](PLACEHOLDER)

// ![Image 33-12](PLACEHOLDER)

// ![Image 33-13](PLACEHOLDER)

// ![Image 33-14](PLACEHOLDER)

// ## Page 34

// TSFM for classification: overview
// Two approaches:
// 1. Large-scale self-supervised pretraining (MOMENT, MANTIS, UniTS, NuTime)
// o Self-supervised means we can use classification or forecasting datasets
// 2. Fine-tuning an LLM (GPT4TS)

// ![Image 34-1](PLACEHOLDER)

// ## Page 35

// TSFM for classification: GPT4TS (2023)
// General idea: make use of LLM backbone
// 1. Project the time-series data to the required dimension of LLM input layer
// 2. Use normalization and patching to make data more homogeneous
// 3. Fine-tune the positional embeddings + layer normalization, keep others fixed

// ![Image 35-1](PLACEHOLDER)

// ## Page 36

// TSFM for classification: GPT4TS (2023)
// A bit oversold: reported accuracy = best epoch test accuracy

// ![Image 36-1](PLACEHOLDER)

// ## Page 37

// TSFM for classification: GPT4TS (2023)
// Some tests on how to fine-tune:
// 1. Slow training with low learning rate
// 2. Freeze LLM, learn only the embedding and head
// 3. Fine-tune all decoder layers instead of 6
// Slow All layers FT 6 layers frozen All layers frozen
// Last epoch +0.03% -2.8% -1% -0.9%
// Best epoch 0% -3.3% -1.6% -1.7%
// x Finetuning the whole GPT2 is worse than freezing it

// ## Page 38

// TSFM for classification: GPT4TS (2023)
// Comparing with other LLMs
// Best of
// GPT2 LLaMA-7b SpeechGPT
// LLMs
// Last epoch 72.3% 70.5% 70.5% 73.8%
// Best epoch 74.8% 72.8% 72.8% 75.9%
// Bigger models ≠ better results

// ## Page 39

// TSFM for classification: MOMENT (2024)
// General idea:
// 1. Pretrain an encoder-only model on a huge pre-training dataset (13M time series)
// 2. Encoder-only = can be used for any downstream task (classification, forecasting, imputation etc)

// ![Image 39-1](PLACEHOLDER)

// ## Page 40

// TSFM for classification: MOMENT (2024)
// o Evidence for classification being dominated by supervised baselines
// o A baseline Resnet beats a foundation model
// o But MOMENT can be used as a solid feature extractor in zero-shot fashion

// ![Image 40-1](PLACEHOLDER)

// ![Image 40-2](PLACEHOLDER)

// ## Page 41

// TSFM for classification: MANTIS (2025)
// General idea:
// 1. Contrastive pre-training on 2M time series, model size = 8M
// 2. Important feature: scalar embedding module for preserving basic statistics

// ![Image 41-1](PLACEHOLDER)


// == Self-supervised and contrastive approaches

// - Autoencoder-based representations
// - Contrastive learning
//   - TS2Vec
//   - CPC, TNC
//   - Temporal augmentations

// - Downstream usage:
//   - Linear probing
//   - Fine-tuning

// Connection to foundation models and transfer learning

// ## Page 42

// Mantis-8M
// Zero-shot performance comparison (UCR+UEA)
//  MANTIS is the best zero-shot FM
// Random Forest
// prediction

// ![Image 42-1](PLACEHOLDER)

// ![Image 42-2](PLACEHOLDER)

// ![Image 42-3](PLACEHOLDER)

// ![Image 42-4](PLACEHOLDER)

// ![Image 42-5](PLACEHOLDER)

// ## Page 43

// Mantis-8M
// Fine-tuning performance comparison
//  MANTIS outperforms other FMs

// ![Image 43-1](PLACEHOLDER)

// ![Image 43-2](PLACEHOLDER)

// ## Page 44

// Mantis-8M
// Fine-tuning performance comparison (different schemes)
//  MANTIS achieves even higher results with early stopping

// ![Image 44-1](PLACEHOLDER)

// ![Image 44-2](PLACEHOLDER)

// ## Page 45

// Conclusions

// ## Page 46

// Take-away messages
// Brief recap:
// o Supervised baselines are very strong
// ! HC2, MultiROCKET et cie are strong contenders
// ! DTW is a powerful tool for TS comparison
// o Transformers from forecasting can be used too!
// ! Yes, you can use PatchTST!
// ! New models from forecasting literature too (SAMformer, iTransformer)
// o And now to foundation models
// ! Less explored than forecasting models, no clear winner
// ! Main advantage: zero-shot performance! (no training/fine-tuning)

// ## Page 47

// Take-away messages
// Practical considerations:
// 1. Always run a baseline comparison with supervised methods
// ! Especially convolution-based ones (ROCKET et cie, TimesNet)
// ! HC2 is very slow but worth trying out too
// 2. Feel free to use famous CV models!
// ! Resnet is still a baseline to beat
// ! Any new CV model (Convnext, Swin) can be applied to TS
// 3. Foundation models are not what they are in NLP and vision
// ! Still a big gap between best supervised baselines and TSFMs
// ! But we are working on it 

// ## Page 48

// Further reading
// Things I’ve omitted in this lecture:
// 1. Self-supervised approaches and their variations
// ! TS2Vec is a very strong self-supervised baseline
// ! Other contenders (T-Loss) exist
// 2. Distionary-based methods (aka bag-of-words)
// 3. Imaging-based time series modelling
// o Like TimesNet but with other ways to represent a 1D TS in 2D
// 4. Tree-based algorithms (Time series forest)
// 5. Automatic feature extraction:
// o Catch22, TS-CHIEFF etc

// ## Page 49

// Thank you.
// Thank you for your attention!
// Noah's Ark Paris
// Contact us
// Noah's Ark Lab Paris

// ![Image 49-1](PLACEHOLDER)

// ![Image 49-2](PLACEHOLDER)

// ## Page 50

// Practical session

// ## Page 51

// Practical session on classification
// Main goals:
// 1. Get hands-on experience with ROCKET (and variations)
// 2. Get hands-on experience with foundation models
//  Mantis + Random forest

// ## Page 52

// Practical session on forecasting
// 1. Supervised classification with random convolutions
//  MultiROCKET
// https://github.com/ChangWeiTan/MultiRocket/tree/main
// 2. Zero-shot classification with foundation models
//  MANTIS-8M
// https://huggingface.co/paris-noah/Mantis-8M

// ## Page 53

// Practical session on classification
// 1. Data
//  UCR datasets in MultiROCKET github
// 2. Step-by-step guide:
//  Use MANTIS repo for feature extraction
// Check tests/test_single_channel_extract_feats.py
//  Use MultiROCKET on this data
//  Compare the results in terms of speed and accuracy
//  Extend to multivariate setup using provided tests in Mantis
// Bonus: combine feature extracted from MultiRocket and Mantis!

// ---

// = Representation Learning for TSC

// == Learning representations before classification

// - Instead of end-to-end classification:
//   - Learn an embedding of the time series
//   - Apply a simple classifier (linear / kNN)

// - Benefits:
//   - Better generalization on small datasets
//   - Transfer across datasets and tasks

// = Data Augmentation for TSC

// == Why augmentation matters

// - Small datasets + deep models
// - Improves robustness and generalization

// Common techniques:
// - Jittering (noise injection)
// - Scaling
// - Time warping
// - Window slicing
// - Mixup for time series

// Often essential for CNN-based approaches

