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

#cetz.canvas({
  import cetz.draw: *

  let d = 5
  let T = 20
  let seed = 0
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
    }
  }

  line((T + 0.25, 2), (T + 1.75, 2), mark: (end: "stealth"), stroke: black + 2pt)

  rect(
    (T + 2.5, 1.75),
    (T + 3, 2.25),
    stroke: newlink,
    fill: newlink,
    radius: 4pt
  )
  
})