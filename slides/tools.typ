#import "cetz/colors.typ": *

#let image-with-caption(image, caption, caption-align: right) = {
  stack(
    spacing: 6pt,
    image,
    align(
      caption-align,
      text(
        size: 0.8em,
        caption,
      ),
    ),
  )
}

#let figure-placeholder(
  width,
  height,
  legend: none,
) = {
  let box = rect(
    width: width,
    height: height,
    stroke: 1pt,
    inset: 0pt,
  )

  if legend == none {
    box
  } else {
    image-with-caption(box, legend)
  }
}

#let grad-disk(
  size: .8em,
  from: xi,
  to: yj,
  angle: 0deg,
) = {
  box(
    width: size,
    height: size,
    baseline: 0%,
  )[
    #let grad = gradient.linear(
      from,
      to,
      angle: angle
    )
    #circle(
      radius: size / 2,
      fill: grad.sharp(2),
    )
  ]
}

// #let block-diag(
//   size: 5cm,
//   blocks: (
//     (1, red),
//     (1, blue),
//     (1, green),
//   ),
//   border: black,
// ) = {
//   let total = blocks.map(b => b.at(0)).sum()
//   let unit = size / total

//   box(
//     width: size,
//     height: size,
//   )[
//     // Outer matrix border
//     #rect(
//       width: size,
//       height: size,
//       stroke: 1pt,
//     )

//     // Diagonal blocks
//     #let offset = 0
//     #for block in blocks {
//       let bsize = block.at(0) * unit
//       let color = block.at(1)

//       rect(
//         x: offset,
//         y: offset,
//         width: bsize,
//         height: bsize,
//         fill: color,
//         stroke: border,
//       )

//       offset += bsize
//     }
//   ]
// }