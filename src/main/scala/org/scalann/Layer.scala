package org.scalann

import breeze.linalg._

abstract class Stage extends Parametrized {

  trait Memo {

    def backward(derivation: DenseVector[Double], outputDeriv: Boolean = false): (DenseVector[Double], DenseVector[Double]) = {
      val inputGrad = DenseVector.zeros[Double](inputSize)
      val paramGrad = DenseVector.zeros[Double](paramSize)

      backwardAdd(derivation, outputDeriv)(inputGrad, paramGrad, 1.0)

      (inputGrad, paramGrad)
    }

    def backwardAdd(derivation: DenseVector[Double], outputDeriv: Boolean = false)(inputGradAcc: DenseVector[Double], paramGradAcc: DenseVector[Double], factor: Double) {
      val (inputGrad, paramGrad) = backward(derivation, outputDeriv)

      inputGradAcc += inputGrad
      paramGradAcc += paramGrad
    }

  }

  def inputSize: Int
  def outputSize: Int
  def paramSize: Int

  /**
   * Stage params as single vector
   */
  def params: DenseVector[Double]

  /**
   * Coefficients for param decay - used to turn off decay of some parameters (such as biases)
   */
  def paramsDecay: DenseVector[Double]

  def apply(input: DenseVector[Double]): DenseVector[Double] =
    forward(input)._1

  def forward(input: DenseVector[Double]): (DenseVector[Double], Memo)

  def updateParams(grad: DenseVector[Double])

  def assignParams(grad: DenseVector[Double])

  def gradientAdd(examples: Traversable[(DenseVector[Double], DenseVector[Double])])(paramGradAcc: DenseVector[Double], factor: Double): Unit =
    examples.foreach { gradientAdd(_)(paramGradAcc, factor / examples.size) }

  def gradient(examples: Traversable[(DenseVector[Double], DenseVector[Double])]): DenseVector[Double] = {
    val res = DenseVector.zeros[Double](paramSize)
    gradientAdd(examples)(res, 1.0)
    res
  }

  def gradient(example: (DenseVector[Double], DenseVector[Double])) = {
    val grad = DenseVector.zeros[Double](paramSize)
    gradientAdd(example)(grad, 1.0)
    grad
  }

  def gradientAdd(example: (DenseVector[Double], DenseVector[Double]))(paramGradientAcc: DenseVector[Double], factor: Double) = {
    val (res, memo) = forward(example._1)
    memo.backwardAdd(res - example._2)(null, paramGradientAcc, factor)
  }

  def exampleLoss(ex: (DenseVector[Double], DenseVector[Double])): Double =
    cost(apply(ex._1), ex._2)

  def cost(actual: DenseVector[Double], target: DenseVector[Double]): Double

  def examplesLoss(examples: Traversable[(DenseVector[Double], DenseVector[Double])]): Double =
    examples.view.map(exampleLoss).sum / examples.size

}
