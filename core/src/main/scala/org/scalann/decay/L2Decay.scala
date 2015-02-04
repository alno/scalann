package org.scalann.decay

import breeze.linalg._

/**
 * L2 weight decay
 */
case class L2Decay(coeff: Double) extends Decay {

  def gradientAdd(params: DenseVector[Double], mask: DenseVector[Double])(gradient: DenseVector[Double], factor: Double) {
    val mult = factor * coeff
    val pd = params.data
    val md = mask.data
    val gd = gradient.data
    var poff = params.offset
    var moff = mask.offset
    var goff = gradient.offset

    var i = 0
    while (i < params.length) {
      gd(goff) -= mult * pd(poff) * md(moff)
      poff += params.stride
      moff += mask.stride
      goff += gradient.stride
      i += 1
    }
  }

}
