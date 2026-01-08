#import "@preview/cetz:0.3.2"
#import "@preview/dice:1.0.0"
#import "colors.typ": *

#let pick_color(seed) = {
  // choose xi or yj at random, apply a random alpha in [0.4, 1.0]
  let (val, seed) = dice.random(seed: seed)
  let base = if val > 0.5 { xi } else { yj }
  let (val, seed) = dice.random(seed: seed)
  let a = 0.4 + 0.6 * val
  return (base.lighten(a * 100%), seed)
}

#let pick_color_label(seed) = {
  let (val, seed) = dice.random(seed: seed)
  let base = if val > 0.5 { newlink } else { oldlink.darken(20%) }
  return (base, seed)
}

#cetz.canvas({
  import cetz.draw: *

  let d = 5
  let T = 20
  let seed = 0
  let seed_label = 0
  let c = none

  for i in range(d) {
    // Draw points
    for t in range(T) {
      (c, seed) = pick_color(seed)
      circle((t, i),
        radius: .3,
        fill: c,
        stroke: c
      )
      if (i == 0) {
        line((t, 5), (t, 6), mark: (end: "stealth"), stroke: black + 2pt)

        (c, seed_label) = pick_color_label(seed_label)
        rect(
          (t - .25, 6.25),
          (t + .25, 6.75),
          stroke: c,
          fill: c,
          radius: 4pt
        )
      }
    }
  }

  
  
})