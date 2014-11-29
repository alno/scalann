package org.scalann

import breeze.linalg._
import org.scalann.loss.Loss

abstract class Stage extends Optimizable[(DenseVector[Double], DenseVector[Double])] with (DenseVector[Double] => DenseVector[Double]) {

  trait Memo {

    def backward(derivation: DenseVector[Double], outputDeriv: Boolean = false): (DenseVector[Double], DenseVector[Double]) = {
      val inputGrad = DenseVector.zeros[Double](inputSize)
      val paramGrad = DenseVector.zeros[Double](paramSize)

      backwardAdd(derivation, outputDeriv)(inputGrad, 1.0, paramGrad, 1.0)

      (inputGrad, paramGrad)
    }

    def backwardAdd(derivation: DenseVector[Double], outputDeriv: Boolean = false)(inputGradAcc: DenseVector[Double], inputFactor: Double, paramGradAcc: DenseVector[Double], paramFactor: Double) {
      val (inputGrad, paramGrad) = backward(derivation, outputDeriv)

      if (inputGradAcc != null)
        inputGradAcc += inputGrad * inputFactor

      if (paramGradAcc != null)
        paramGradAcc += paramGrad * paramFactor
    }

  }

  def inputSize: Int
  def outputSize: Int

  def apply(input: DenseVector[Double]): DenseVector[Double] =
    forward(input)._1

  def forward(input: DenseVector[Double]): (DenseVector[Double], Memo)

  def gradientAdd(example: (DenseVector[Double], DenseVector[Double]))(paramGradientAcc: DenseVector[Double], factor: Double) = {
    val (res, memo) = forward(example._1)
    memo.backwardAdd(res - example._2)(null, Double.NaN, paramGradientAcc, factor)
  }

  def exampleLoss(ex: (DenseVector[Double], DenseVector[Double])): Double =
    loss(apply(ex._1), ex._2)

  def examplesLoss(examples: Traversable[(DenseVector[Double], DenseVector[Double])]): Double =
    examples.view.map(exampleLoss).sum / examples.size

  def loss: Loss

}
