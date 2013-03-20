package org.scalann

import breeze.linalg._
import scala.math.exp

class SoftmaxLayer(inputSize: Int, outputSize: Int) extends AbstractLayer(inputSize, outputSize) {

  private[this] val tiny = 1e-300

  protected def outputTransform(v: DenseVector[Double]) = {
    val data = v.data
    val stride = v.stride
    val max = v.max

    var pos = v.offset
    var ind = 0
    var sum = 0.0

    while (ind < v.size) {
      val cur = exp(data(pos) - max)

      data(pos) = cur
      sum += cur
      pos += stride
      ind += 1
    }

    v /= sum
  }

  protected def outputDerivationTransform(dv: DenseVector[Double], v: DenseVector[Double]) {}

  def cost(actual: DenseVector[Double], target: DenseVector[Double]): Double =
    -(actual.activeValuesIterator zip target.activeValuesIterator).map {
      case (a, b) => math.log(a + tiny) * b
    }.sum
}