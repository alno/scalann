package org.scalann.decay

import breeze.linalg._

/**
 * No weight decay - NOP implementation
 */
object NoDecay extends Decay {

  def gradientAdd(params: DenseVector[Double], mask: DenseVector[Double])(gradient: DenseVector[Double], factor: Double) {}

}
