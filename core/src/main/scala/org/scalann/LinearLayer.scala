package org.scalann

import breeze.linalg._

class LinearLayer(inputSize: Int, outputSize: Int) extends AbstractLayer(inputSize, outputSize) {

  protected def outputTransform(v: DenseVector[Double]) {}

  protected def outputDerivationTransform(dv: DenseVector[Double], v: DenseVector[Double]) {}

  def cost(actual: DenseVector[Double], target: DenseVector[Double]): Double =
    0.5 * (actual.activeValuesIterator zip target.activeValuesIterator).map {
      case (a, b) => (a - b) * (a - b)
    }.sum

}
