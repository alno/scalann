package org.scalann

import breeze.linalg._

abstract class BasicLayer(val inputSize: Int, val outputSize: Int) extends Stage {

  val paramSize = outputSize * (inputSize + 1)

  private[this] val params = DenseVector.fill(paramSize) { math.random * 2 - 1 }

  private[this] val weights: DenseMatrix[Double] = new DenseMatrix(outputSize, inputSize, params.data)
  private[this] val biases: DenseVector[Double] = new DenseVector(params.data, outputSize * inputSize, 1, outputSize)

  def forward(input: DenseVector[Double]) = {
    val result = outputTransform(weights * input + biases)

    result -> new Memo {

      def backward(derivation: DenseVector[Double]) = {
        val resData = new Array[Double](paramSize)

        val dEh = derivation :* outputDerivation(result) // Derivation by activations

        val dEx = weights.t * dEh // Derivation by inputs

        val dEw = dEh * input.t // Derivation by weights
        val dEb = dEh // Derivation by biases

        new DenseMatrix(outputSize, inputSize, resData) := dEw
        new DenseVector(resData, outputSize * inputSize, 1, outputSize) := dEb

        dEx -> new DenseVector(resData, 0, 1, paramSize)
      }

    }
  }

  def update(gradient: DenseVector[Double]) =
    params += gradient

  protected def outputTransform(xv: DenseVector[Double]): DenseVector[Double]

  protected def outputDerivation(yv: DenseVector[Double]): DenseVector[Double]

}

class LogisticLayer(inputSize: Int, outputSize: Int) extends BasicLayer(inputSize, outputSize) {

  protected def outputTransform(xv: DenseVector[Double]) =
    xv.map { x => 1 / (1 + math.exp(-x)) }

  protected def outputDerivation(yv: DenseVector[Double]) =
    yv.map { y => y * (1 - y) }

}

class SoftmaxLayer(inputSize: Int, outputSize: Int) extends BasicLayer(inputSize, outputSize) {

  protected def outputTransform(xv: DenseVector[Double]) = {
    val exps = xv.map(math.exp)
    val expSum = exps.sum

    exps.map { _ / expSum }
  }

  protected def outputDerivation(yv: DenseVector[Double]) =
    yv

}