#import "@preview/touying:0.6.1": *
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning for Time Series - Basics],
  config-info(
    title: [Deep Learning for Time Series],
    subtitle: [Session 0: Administrative information],
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

== Administrative information

- Teacher: Romain Tavenard
  - Affiliation: IRISA-Inria, Univ. Rennes
  - Research interests: time series analysis, machine learning, optimal transport
    - Interested in an internship or PhD on these topics? Contact me!
  - Email: romain.tavenard\@univ-rennes2.fr
- Course organization
  - Mixed Lectures-Lab sessions: Mondays 09:15-12:45
  - Duration: 6 weeks
  - Evaluation: project + report (more details on those later)