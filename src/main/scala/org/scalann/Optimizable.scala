package org.scalann

import breeze.linalg._

trait Optimizable[T] extends Parametrized {

  def gradientAdd(example: T)(paramGradAcc: DenseVector[Double], factor: Double)

  def gradient(example: T): DenseVector[Double] = {
    val res = DenseVector.zeros[Double](paramSize)
    gradientAdd(example)(res, 1.0)
    res
  }

  def gradientAdd(examples: Traversable[T])(paramGradAcc: DenseVector[Double], factor: Double): Unit =
    examples.foreach { gradientAdd(_)(paramGradAcc, factor / examples.size) }

  def gradient(examples: Traversable[T]): DenseVector[Double] = {
    val res = DenseVector.zeros[Double](paramSize)
    gradientAdd(examples)(res, 1.0)
    res
  }

}
