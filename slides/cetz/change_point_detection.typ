#import "@preview/cetz:0.3.2"
#import "@preview/dice:1.0.0"
#import "colors.typ": *

#let pick_color_blue(seed) = {
  // choose xi or yj at random, apply a random alpha in [0.4, 1.0]
  let base = xi
  let (val, seed) = dice.random(seed: seed)
  let a = 0.4 + 0.6 * val
  return (base.lighten(a * 100%), seed)
}

#let pick_color_purple(seed) = {
  // choose xi or yj at random, apply a random alpha in [0.4, 1.0]
  let base = yj
  let (val, seed) = dice.random(seed: seed)
  let a = 0.4 + 0.6 * val
  return (base.lighten(a * 100%), seed)
}

#cetz.canvas({
  import cetz.draw: *

  let d = 5
  let T = 20
  let T_break = 12
  let T_break_back = 17
  let seed = 0
  let state = false
  let seed_label = 0
  let c = none

  for i in range(d) {
    // Draw points
    for t in range(T) {
      if (t < T_break or t >= T_break_back) {
        (c, seed) = pick_color_blue(seed)
      } else {
        (c, seed) = pick_color_purple(seed)
      }
      circle((t, i),
        radius: .3,
        fill: c,
        stroke: c
      )
    }
  }

  for t in (T_break, T_break_back) {
    line((t - .5, -0.5), (t - .5, 4.5), 
         stroke: (paint: black, thickness: 1.5pt, dash: "dashed"))
  }

})