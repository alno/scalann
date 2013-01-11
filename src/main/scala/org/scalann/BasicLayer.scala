package org.scalann

import breeze.linalg._
import scala.math._
import org.netlib.blas.Dgemv
import org.netlib.blas.Dgemm
import org.netlib.blas.Daxpy

abstract class BasicLayer(val inputSize: Int, val outputSize: Int) extends Stage {

  val paramSize = outputSize * (inputSize + 1)

  val params = DenseVector.fill(paramSize) { math.random * 2 - 1 }

  private[this] val weights: DenseMatrix[Double] = new DenseMatrix(outputSize, inputSize, params.data)
  private[this] val biases: DenseVector[Double] = new DenseVector(params.data, outputSize * inputSize, 1, outputSize)

  def forward(input: DenseVector[Double]) = {
    val result = weights * input

    result += biases

    outputTransform(result)

    result -> new Memo {

      override def backwardAdd(derivation: DenseVector[Double], outputDeriv: Boolean)(inputGradAcc: DenseVector[Double], paramGradAcc: DenseVector[Double]) {
        require(inputGradAcc.size == inputSize)
        require(paramGradAcc.size == paramSize)
        require(paramGradAcc.stride == 1)

        var dEh = derivation // Derivation by internal activations

        if (outputDeriv) { // If output derivation should be transformed
          dEh = dEh.copy // Copy it
          outputDerivationTransform(dEh, result) // Transform (destructively)
        }

        // inputGradAcc += weights.t * dEh // Append derivation by inputs
        Dgemv.dgemv("t", outputSize, inputSize,
          1.0, weights.data, weights.offset, weights.majorStride,
          dEh.data, dEh.offset, 1,
          1.0, inputGradAcc.data, inputGradAcc.offset, inputGradAcc.stride)

        // paramGradAcc(dEw) += dEh * input.t // Derivation by weights
        Dgemm.dgemm("n", "t", outputSize, inputSize, 1,
          1.0, dEh.data, dEh.offset, outputSize,
          input.data, input.offset, inputSize,
          1.0, paramGradAcc.data, paramGradAcc.offset, outputSize)

        // paramGradAcc(dEb) += dEh // Derivation by biases
        Daxpy.daxpy(outputSize, 1.0,
          dEh.data, dEh.offset, 1,
          paramGradAcc.data, paramGradAcc.offset + outputSize * inputSize, 1);
      }

    }
  }

  def update(gradient: DenseVector[Double]) =
    params += gradient

  protected def outputTransform(v: DenseVector[Double])

  protected def outputDerivationTransform(dv: DenseVector[Double], v: DenseVector[Double])

}

class LinearLayer(inputSize: Int, outputSize: Int) extends BasicLayer(inputSize, outputSize) {

  protected def outputTransform(v: DenseVector[Double]) {}

  protected def outputDerivationTransform(dv: DenseVector[Double], v: DenseVector[Double]) {}

  def cost(actual: DenseVector[Double], target: DenseVector[Double]): Double =
    0.5 * (actual.activeValuesIterator zip target.activeValuesIterator).map {
      case (a, b) => (a - b) * (a - b)
    }.sum

}

class LogisticLayer(inputSize: Int, outputSize: Int) extends BasicLayer(inputSize, outputSize) {

  private[this] val tiny = 1e-300

  protected def outputTransform(v: DenseVector[Double]) {
    val data = v.data
    val stride = v.stride

    var pos = v.offset
    var ind = 0

    while (ind < v.size) {
      data(pos) = 1 / (1 + exp(-data(pos)))
      pos += stride
      ind += 1
    }
  }

  protected def outputDerivationTransform(dv: DenseVector[Double], v: DenseVector[Double]) {
    val ddata = dv.data
    val dstride = dv.stride

    val vdata = v.data
    val vstride = v.stride

    var dpos = dv.offset
    var vpos = v.offset
    var ind = 0

    while (ind < dv.size) {
      ddata(dpos) *= vdata(vpos) * (1 - vdata(vpos))
      dpos += dstride
      vpos += vstride
      ind += 1
    }
  }

  def cost(actual: DenseVector[Double], target: DenseVector[Double]): Double =
    -(actual.activeValuesIterator zip target.activeValuesIterator).map {
      case (a, b) => math.log(a + tiny) * b + math.log(1 - a + tiny) * (1 - b)
    }.sum

}

class SoftmaxLayer(inputSize: Int, outputSize: Int) extends BasicLayer(inputSize, outputSize) {

  private[this] val tiny = 1e-300

  protected def outputTransform(v: DenseVector[Double]) = {
    val data = v.data
    val stride = v.stride
    val max = v.max

    var pos = v.offset
    var ind = 0
    var sum = 0.0

    while (ind < v.size) {
      val cur = exp(data(pos) - max)

      data(pos) = cur
      sum += cur
      pos += stride
      ind += 1
    }

    v /= sum
  }

  protected def outputDerivationTransform(dv: DenseVector[Double], v: DenseVector[Double]) {}

  def cost(actual: DenseVector[Double], target: DenseVector[Double]): Double =
    -(actual.activeValuesIterator zip target.activeValuesIterator).map {
      case (a, b) => math.log(a + tiny) * b
    }.sum
}