package org.scalann.activation

import breeze.linalg._
import breeze.numerics._

object Softmax extends ActivationTransform {

  override def transformOutput(v: DenseVector[Double]): Unit = {
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

  override def transformOutputDerivation(dv: DenseVector[Double], v: DenseVector[Double]): Unit = ???

}
