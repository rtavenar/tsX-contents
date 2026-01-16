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

  // Draw points
  for t in range(T) {
    (c, seed) = pick_color(seed)
    circle((t, 0),
      radius: .3,
      fill: c,
      stroke: c,
      name: "x_" + str(t)
    )
  }
  content((-1, 0.1), text(size: 16pt)[...])
  content((T, 0.1), text(size: 16pt)[...])

  // Draw filter
  for t in range(3) {
    (c, seed) = pick_color(seed)
    circle((1.5 + t, 2),
      radius: .3,
      fill: c,
      stroke: c,
      name: "f_" + str(t)
    )
  }
  rect((1., 1.5), (4, 2.5), radius: 3pt, name: "filter")
  content((0.2, 2), text(size: 16pt)[Filter])


  let pos = 6
  content((pos, -1.0), text(newlink, size: 28pt, [$x_t$]))
  rect((pos - 1.5, - 1.7), (pos + 1.5, .5), radius: 3pt, name: "subseries")

  // Draw points
  for t in range(T) {
    (c, seed) = pick_color(seed)
    circle((t, 4),
      radius: .3,
      fill: c,
      stroke: c,
      name: "o_" + str(t)
    )
  }
  content((pos, 5.0), text(newlink, size: 28pt, [$o_t$]))
  content((-1, 4.1), text(size: 16pt)[...])
  content((T, 4.1), text(size: 16pt)[...])

  circle((pos, 2),
    radius: .6,
    stroke: black,
    name: "dot-circle"
  )
  content((pos, 2), text(size: 16pt, [dot]))

  line("subseries.north", "dot-circle.south", mark: (end: ">"))

  line("filter.east", "dot-circle.west", mark: (end: ">"))

  line("dot-circle.north", "o_6.south", mark: (end: ">"))


})