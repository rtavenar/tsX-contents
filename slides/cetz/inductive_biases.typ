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

  let T = 5
  let target_t = 3
  let delta_x = 1.2
  let radius = 0.3

  // ConvNets
  content((T / 2, 3), text(oldlink.darken(40%), size: 24pt, [Causal Conv.]))
  for t in range(T) {
    circle((t * delta_x, 0),
      radius: radius,
      fill: xi,
      stroke: black,
      stroke-width: 1pt,
      name: "conv_x_" + str(t)
    )
    let color = if t == target_t { yj } else { yj.lighten(60%) }
    circle((t * delta_x, 1.5),
      radius: radius,
      fill: color,
      stroke: black,
      stroke-width: 1pt,
      name: "conv_y_" + str(t)
    )
  }
  for t in range(target_t - 2, target_t + 1) {
    line("conv_x_" + str(t) + ".north", "conv_y_3.south", mark: (end: "stealth"), stroke: black + 2pt)
  }

  // RNNs
  let x_shift = T * delta_x + 2
  content((x_shift + T / 2, 3), text(oldlink.darken(40%), size: 24pt, [RNN]))
  for t in range(T) {
    circle((t * delta_x + x_shift, 0),
      radius: radius,
      fill: xi,
      stroke: black,
      stroke-width: 1pt,
      name: "rnn_x_" + str(t)
    )
    let color = if t == target_t { yj } else { yj.lighten(60%) }
    circle((t * delta_x + x_shift, 1.5),
      radius: radius,
      fill: color,
      stroke: black,
      stroke-width: 1pt,
      name: "rnn_y_" + str(t)
    )
  }
  for t in range(target_t) {
    line("rnn_y_" + str(t) + ".east", "rnn_y_" + str(t + 1) + ".west", mark: (end: "stealth"), stroke: (paint: black, thickness: 2pt, dash: "dashed"))
    line("rnn_x_" + str(t) + ".north", "rnn_y_" + str(t) + ".south", mark: (end: "stealth"), stroke: (paint: black, thickness: 2pt, dash: "dashed"))
  }
  line("rnn_x_" + str(target_t) + ".north", "rnn_y_" + str(target_t) + ".south", mark: (end: "stealth"), stroke: black + 2pt)

  // Self-Attention
  x_shift = 2 * x_shift
  content((x_shift + T / 2, 3), text(oldlink.darken(40%), size: 24pt, [Self-Attention]))
  for t in range(T) {
    circle((t * delta_x + x_shift, 0),
      radius: radius,
      fill: xi,
      stroke: black,
      stroke-width: 1pt,
      name: "sa_x_" + str(t)
    )
    let color = if t == target_t { yj } else { yj.lighten(60%) }
    circle((t * delta_x + x_shift, 1.5),
      radius: radius,
      fill: color,
      stroke: black,
      stroke-width: 1pt,
      name: "sa_y_" + str(t)
    )
  }
  for t in range(T) {
    line("sa_x_" + str(t) + ".north", "sa_y_3.south", mark: (end: "stealth"), stroke: black + 2pt)
  }

  // rect(
  //   (14 - 0.44, -0.5),
  //   (T - 0.5, d - .5),
  //   stroke: oldlink.darken(20%),
  //   radius: 4pt
  // )
  // content((16.5, -1.0), text(oldlink.darken(20%), size: 24pt, [_Horizon_]))

  // rect(
  //   (-0.5, -0.5),
  //   (14 - 0.56, d - .5),
  //   stroke: newlink,
  //   radius: 4pt
  // )
  // content((6.5, -1.0), text(newlink, size: 24pt, [Past window]))
  
})