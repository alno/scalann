package org.scalann

import breeze.linalg._

class BasicLayer(val inputSize: Int, val outputSize: Int) extends Stage {

  private[this] val weights: DenseMatrix[Double] = DenseMatrix.fill(outputSize, inputSize) { math.random * 2 - 1 }
  private[this] val biases: DenseVector[Double] = DenseVector.fill(outputSize) { math.random * 2 - 1 }

  def forward(input: DenseVector[Double]) = {
    val result = outputTransform(weights * input + biases)

    result -> new Memo {

      def backward(derivation: DenseVector[Double]) = {
        val dEh = derivation :* outputDerivation(result) // Derivation by activations

        val dEx = weights.t * dEh // Derivation by inputs

        val dEw = dEh * input.t // Derivation by weights
        val dEb = dEh // Derivation by biases

        dEx -> WeightBiasGradient(dEw, dEb)
      }

    }
  }

  def update(grad: Gradient) = grad match {
    case WeightBiasGradient(weightGradient, biasGradient) =>
      weights += weightGradient
      biases += biasGradient
  }

  protected def outputTransform(xv: DenseVector[Double]) =
    xv.map { x => 1 / (1 + math.exp(-x)) }

  protected def outputDerivation(yv: DenseVector[Double]) =
    yv.map { y => y * (1 - y) }

}
