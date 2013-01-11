package org.scalann

import breeze.linalg._

abstract class Stage {

  trait Memo {

    def layer: Stage

    def backward(derivation: DenseVector[Double], outputDeriv: Boolean = false): (DenseVector[Double], DenseVector[Double]) = {
      val inputGrad = DenseVector.zeros[Double](inputSize)
      val paramGrad = DenseVector.zeros[Double](paramSize)

      backwardAdd(derivation, outputDeriv)(inputGrad, paramGrad)

      (inputGrad, paramGrad)
    }

    def backwardAdd(derivation: DenseVector[Double], outputDeriv: Boolean)(inputGradAcc: DenseVector[Double], paramGradAcc: DenseVector[Double]) {
      val (inputGrad, paramGrad) = backward(derivation, outputDeriv)
      
      inputGradAcc += inputGrad
      paramGradAcc += paramGrad
    }
    
  }

  def inputSize: Int
  def outputSize: Int
  def paramSize: Int

  def params: DenseVector[Double]

  def apply(input: DenseVector[Double]): DenseVector[Double] =
    forward(input)._1

  def forward(input: DenseVector[Double]): (DenseVector[Double], Memo)

  def update(grad: DenseVector[Double])

  def examplesGradient(examples: Traversable[(DenseVector[Double], DenseVector[Double])]) = {
    val grad = exampleGradient(examples.head)

    examples.tail.foreach { ex =>
      grad += exampleGradient(ex)
    }

    grad *= 1.0 / examples.size
    grad
  }

  def exampleGradient(example: (DenseVector[Double], DenseVector[Double])) = {
    val (res, memo) = forward(example._1)
    memo.backward(res - example._2)._2
  }

  def exampleLoss(ex: (DenseVector[Double], DenseVector[Double])): Double =
    cost(apply(ex._1), ex._2)

  def cost(actual: DenseVector[Double], target: DenseVector[Double]): Double

  def examplesLoss(examples: Traversable[(DenseVector[Double], DenseVector[Double])]): Double =
    examples.view.map(exampleLoss).sum / examples.size

}
