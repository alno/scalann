package org.scalann

import breeze.linalg._

abstract class BasicLayer(val inputSize: Int, val outputSize: Int) extends Stage {

  val paramSize = outputSize * (inputSize + 1)

  val params = DenseVector.fill(paramSize) { math.random * 2 - 1 }

  private[this] val weights: DenseMatrix[Double] = new DenseMatrix(outputSize, inputSize, params.data)
  private[this] val biases: DenseVector[Double] = new DenseVector(params.data, outputSize * inputSize, 1, outputSize)

  def forward(input: DenseVector[Double]) = {
    val result = outputTransform(weights * input + biases)

    //println(result)

    result -> new Memo {

      def backward(derivation: DenseVector[Double], outputDeriv: Boolean = false) = {
        val resData = new Array[Double](paramSize)

        val dEh = if (outputDeriv)
          derivation :* outputDerivation(result) // Derivation by activations
        else
          derivation

        val dEx = weights.t * dEh // Derivation by inputs

        val dEw = dEh * input.t // Derivation by weights
        val dEb = dEh // Derivation by biases

        // println(weights, dEx)

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

class LinearLayer(inputSize: Int, outputSize: Int) extends BasicLayer(inputSize, outputSize) {

  protected def outputTransform(xv: DenseVector[Double]) =
    xv

  protected def outputDerivation(yv: DenseVector[Double]) =
    DenseVector.fill(yv.size)(1.0)

  def cost(actual: DenseVector[Double], target: DenseVector[Double]): Double =
    0.5 * (actual.activeValuesIterator zip target.activeValuesIterator).map {
      case (a, b) => (a - b) * (a - b)
    }.sum

}

class LogisticLayer(inputSize: Int, outputSize: Int) extends BasicLayer(inputSize, outputSize) {

  private[this] val tiny = 1e-300

  protected def outputTransform(xv: DenseVector[Double]) =
    xv.map { x => 1 / (1 + math.exp(-x)) }

  protected def outputDerivation(yv: DenseVector[Double]) =
    yv.map { y => y * (1 - y) }

  def cost(actual: DenseVector[Double], target: DenseVector[Double]): Double =
    -(actual.activeValuesIterator zip target.activeValuesIterator).map {
      case (a, b) => math.log(a + tiny) * b + math.log(1 - a + tiny) * (1 - b)
    }.sum

}

class SoftmaxLayer(inputSize: Int, outputSize: Int) extends BasicLayer(inputSize, outputSize) {

  private[this] val tiny = 1e-300

  protected def outputTransform(xv: DenseVector[Double]) = {
    val m = xv.max
    val exps = xv.map { x => math.exp(x - m) }
    val expSum = exps.sum

    exps.map { _ / expSum }
  }

  protected def outputDerivation(yv: DenseVector[Double]) =
    DenseVector.fill(yv.size)(1.0)

  def cost(actual: DenseVector[Double], target: DenseVector[Double]): Double =
    -(actual.activeValuesIterator zip target.activeValuesIterator).map {
      case (a, b) => math.log(a + tiny) * b
    }.sum
}