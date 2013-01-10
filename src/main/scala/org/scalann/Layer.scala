package org.scalann

import breeze.linalg._

abstract class Stage {

  trait Memo {

    def backward(derivation: DenseVector[Double]): (DenseVector[Double], DenseVector[Double])

  }

  def inputSize: Int
  def outputSize: Int
  def paramSize: Int

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

  def examplesLoss(examples: Traversable[(DenseVector[Double], DenseVector[Double])]): Double =
    -examples.view.map { ex =>
      (apply(ex._1).activeValuesIterator zip ex._2.activeValuesIterator).map {
        case (a, b) => math.log(a) * b
      }.sum
    }.sum / examples.size

}
