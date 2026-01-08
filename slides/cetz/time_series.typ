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

  rect(
    (-1., -2.0),
    (T, d),
    stroke: oldlink,
    radius: 4pt
  )

  content((0., -1.), text(newlink, size: 28pt, [$x_1$]))
  content((12.0, -1.0), text(newlink, size: 28pt, [$x_t$]))
  content((T - 1.0, -1.0), text(newlink, size: 28pt, [$x_T$]))
  for pos in (0., 12., T - 1.0) {
      rect(
      (pos - .5, -0.5),
      (pos + .5, d - .5),
      stroke: newlink,
      radius: 4pt
    )
  }
  content((T + 0.6, -1.5), text(oldlink, size: 32pt, [$X$]))
})