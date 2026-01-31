#import "@preview/touying:0.6.1": *
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning for Time Series - Basics],
  config-info(
    title: [Deep Learning for Time Series],
    subtitle: [Details on the project],
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

== Project

- Task: Time Series Classification (Dataset: #link("https://timeseriesclassification.com/description.php?Dataset=LSST")[LSST])
- Choose one of the following settings
  - Setting 1: Adapt a foundation model
  - Setting 2: Pre-train on forecasting (Informer datasets only), adapt on classification

---

- Requirements
  - A group = 3 or 4 students
  - A strong competitor + a decent baseline (without pre-training)
  - A report is due on March 16th, 23:59 Paris time: max 3 pages #link("https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip")[ICML 2026 style] (incl. a link to a git repo for the code)
  - Defense: 10 minutes presentation + 5 minutes question, on the 23rd of March
    - Each student should attend the session her/his project is assigned (morning or afternoon)

---

== Data loading

- You can use `tslearn` to load the LSST dataset as `numpy` arrays:

#set text(size:24pt)

```python
from tslearn.datasets import UCR_UEA_datasets

# Load the LSST dataset from UEA archive
ds = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ds.load_dataset("LSST")
```