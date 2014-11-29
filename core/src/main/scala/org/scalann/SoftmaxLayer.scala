package org.scalann

import breeze.linalg._
import scala.math.exp
import org.scalann.loss.SoftmaxLoss

class SoftmaxLayer(inputSize: Int, outputSize: Int) extends AbstractLayer(inputSize, outputSize) {

  protected def outputTransform(v: DenseVector[Double]) = {
    val data = v.data
    val stride = v.stride
    val maxVal = max(v)

    var pos = v.offset
    var ind = 0
    var sum = 0.0

    while (ind < v.size) {
      val cur = exp(data(pos) - maxVal)

      data(pos) = cur
      sum += cur
      pos += stride
      ind += 1
    }

    v /= sum
  }

  protected def outputDerivationTransform(dv: DenseVector[Double], v: DenseVector[Double]) {}

  override def loss = SoftmaxLoss

}
