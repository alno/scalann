package org.scalann.loss

import breeze.linalg.Vector

object LogisticLoss extends Loss {

  private[this] val tiny = 1e-300

  override def apply(actual: Vector[Double], target: Vector[Double]): Double =
    -(actual.activeValuesIterator zip target.activeValuesIterator).map {
      case (a, b) => math.log(a + tiny) * b + math.log(1 - a + tiny) * (1 - b)
    }.sum

}
