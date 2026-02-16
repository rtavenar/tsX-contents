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

  let T = 10
  let seed = 0
  let c = none
  let pos = 6

  // Draw points
  for t in range(T) {
    (c, seed) = pick_color(seed)
    circle((t, 0),
      radius: .3,
      fill: c,
      stroke: c,
      name: "x_" + str(t)
    )
    if (t < pos) {
      content((t, 0.), text(size: 10pt)[$x_(t-#(10 - t - 4))$])
    } else if (t == pos) {
      content((t, 0.), text(size: 10pt)[$x_t$])
    }
  }
  content((T, 0.1), text(size: 16pt)[...])

  // Draw filter
  for t in range(10) {
    (c, seed) = pick_color(seed)
    circle((1. + t - 4, 1),
      radius: .3,
      fill: c,
      stroke: c,
      name: "f_" + str(t)
    )
    if (t < 9) {
      content((1. + t - 4, 1), text(size: 10pt)[$overline(A)^#(10 - t - 1) overline(B)$])
    } else {
      content((1. + t - 4, 1), text(size: 10pt)[$overline(B)$])
    }
  }
  rect((-3.5, 0.6), (6.5, 1.4), radius: 3pt, name: "filter")
  content((-5, 1), text(size: 16pt)[Filter])
  content((-5, 0), text(size: 16pt)[Time series])


  // content((pos, -1.0), text(newlink, size: 28pt, [$x_t$]))
  rect((pos - 6.5, - .4), (pos + .5, .4), radius: 3pt, name: "subseries")

  // Draw points
  // for t in range(T) {
  //   (c, seed) = pick_color(seed)
  //   circle((t, 4),
  //     radius: .3,
  //     fill: c,
  //     stroke: c,
  //     name: "o_" + str(t)
  //   )
  // }
  // content((pos, 5.0), text(newlink, size: 28pt, [$o_t$]))
  // content((-1, 4.1), text(size: 16pt)[...])
  // content((T, 4.1), text(size: 16pt)[...])

  // circle((pos, 2),
  //   radius: .6,
  //   stroke: black,
  //   name: "dot-circle"
  // )
  // content((pos, 2), text(size: 16pt, [dot]))

  // line("subseries.north", "dot-circle.south-west", mark: (end: ">"))

  // line("filter.east", "dot-circle.west", mark: (end: ">"))

  // line("dot-circle.north", "o_6.south", mark: (end: ">"))


})