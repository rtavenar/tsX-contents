#import "@preview/cetz:0.3.2"
#import "@preview/dice:1.0.0"
#import "colors.typ": *

#let pick_color(seed, base) = {
  // choose xi or yj at random, apply a random alpha in [0.4, 1.0]
  let (val, seed) = dice.random(seed: seed)
  let (val, seed) = dice.random(seed: seed)
  let a = 0.4 + 0.6 * val
  return (base.lighten(a * 100%), seed)
}

#cetz.canvas({
  import cetz.draw: *

  let d = 3
  let T = 20
  let T_past = 14
  let seed = 0
  let c = none

  for i in range(d) {
    // Draw points
    for t in range(T_past) {
      (c, seed) = pick_color(seed, xi)
      circle((t, i),
        radius: .3,
        fill: c,
        stroke: c
      )
    }
  }

  line((T_past - 1, d - .5), (T_past - 1, d + .5), mark: (end: "stealth"), stroke: black + 2pt)

  for i in range(d) {
    // Draw points
    for t in range(T_past, T_past + 2) {
      (c, seed) = pick_color(seed, yj)
      circle((t - 1, i + d + 1),
        radius: .3,
        fill: c,
        stroke: c
      )
      if t == T_past {
        circle((t, i),
          radius: .3,
          fill: c,
          stroke: c
        )
      }
    }
  }
  rect(
    (T_past - 1.5, 2 * d + 0.5),
    (T - 1.5, d + .5),
    stroke: oldlink,
    radius: 4pt
  )

  line((T_past - .9, d + .5), (T_past - .1, d - .5), mark: (end: "stealth"), stroke: (paint: yj, thickness: 2pt, dash: "dashed"))

  line((T_past, d - .5), (T_past, d + .5), mark: (end: "stealth"), stroke: black + 2pt)

  // Legend
  line((5, d + 3), (6, d + 3), mark: (end: "stealth"), stroke: black + 2pt)
  content((6.5, d + 3), anchor: "west", text([Model prediction], size: 18pt))
  line((5, d + 2), (6, d + 2), mark: (end: "stealth"), stroke: (paint: yj, thickness: 2pt, dash: "dashed"))
  content((6.5, d + 2), anchor: "west", text([Copy], size: 18pt, fill: yj))
  
})