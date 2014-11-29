package org.scalann.loss

import breeze.linalg.Vector

object SoftmaxLoss extends Loss {

  private[this] val tiny = 1e-300

  override def apply(actual: Vector[Double], target: Vector[Double]): Double =
    -(actual.activeValuesIterator zip target.activeValuesIterator).map {
      case (a, b) => math.log(a + tiny) * b
    }.sum

}
