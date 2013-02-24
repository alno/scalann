package org.scalann.decay

import breeze.linalg._

trait Decay {

  def gradientAdd(params: DenseVector[Double], mask: DenseVector[Double])(gradient: DenseVector[Double], coeff: Double)

}
