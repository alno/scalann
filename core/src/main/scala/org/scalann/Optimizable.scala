package org.scalann

import breeze.linalg._

trait Optimizable[T] extends Parametrized {

  type Example = T

  def gradientAdd(example: Example)(paramGradAcc: DenseVector[Double], factor: Double)

  def gradient(example: Example): DenseVector[Double] = {
    val res = DenseVector.zeros[Double](paramSize)
    gradientAdd(example)(res, 1.0)
    res
  }

  def gradientAdd(examples: Traversable[Example])(paramGradAcc: DenseVector[Double], factor: Double): Unit =
    examples.foreach { gradientAdd(_)(paramGradAcc, factor / examples.size) }

  def gradient(examples: Traversable[Example]): DenseVector[Double] = {
    val res = DenseVector.zeros[Double](paramSize)
    gradientAdd(examples)(res, 1.0)
    res
  }

}
