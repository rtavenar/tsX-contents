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
  - A strong competitor + a decent baseline (without pre-training)
  - A report is due on March 16th, 23:59 Paris time: max 3 pages #link("https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip")[ICML 2026 style] (incl. a link to a git repo for the code)
  - Defense: 10 minutes presentation + 5 minutes question
- Each student should attend the session her/his project is assigned (morning or afternoon)