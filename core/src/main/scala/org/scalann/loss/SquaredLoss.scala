package org.scalann.loss

import breeze.linalg.Vector

object SquaredLoss extends Loss {

  override def apply(actual: Vector[Double], target: Vector[Double]): Double =
    0.5 * (actual.activeValuesIterator zip target.activeValuesIterator).map {
      case (a, b) => (a - b) * (a - b)
    }.sum

}
