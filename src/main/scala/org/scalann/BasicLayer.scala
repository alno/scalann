package org.scalann

import breeze.linalg._
import breeze.numerics._
import scala.math.exp

import org.netlib.blas.Dgemv
import org.netlib.blas.Daxpy
import org.netlib.blas.Dger

abstract class BasicLayer(val inputSize: Int, val outputSize: Int) extends Stage {

  val paramSize = outputSize * (inputSize + 1)

  val params = DenseVector.fill(paramSize) { math.random * 2 - 1 }

  val weights: DenseMatrix[Double] = new DenseMatrix(outputSize, inputSize, params.data)
  val biases: DenseVector[Double] = new DenseVector(params.data, outputSize * inputSize, 1, outputSize)

  val paramsDecay =
    DenseVector.vertcat(DenseVector.fill(outputSize * inputSize)(1.0), DenseVector.fill(outputSize)(0.0))

  def forward(input: DenseVector[Double]) = {
    val result = weights * input
    result += biases
    outputTransform(result)

    result -> new Memo {

      override def backwardAdd(derivation: DenseVector[Double], outputDeriv: Boolean)(inputGradAcc: DenseVector[Double], inputFactor: Double, paramGradAcc: DenseVector[Double], paramFactor: Double) {
        var dEh = derivation // Derivation by internal activations

        if (outputDeriv) { // If output derivation should be transformed
          dEh = dEh.copy // Copy it
          outputDerivationTransform(dEh, result) // Transform (destructively)
        }

        if (inputGradAcc != null) {
          require(inputGradAcc.size == inputSize)

          // inputGradAcc += weights.t * dEh * factor// Append derivation by inputs
          Dgemv.dgemv("t", outputSize, inputSize,
            inputFactor, weights.data, weights.offset, weights.majorStride,
            dEh.data, dEh.offset, dEh.stride,
            1.0, inputGradAcc.data, inputGradAcc.offset, inputGradAcc.stride)
        }

        if (paramGradAcc != null) {
          require(paramGradAcc.size == paramSize)
          require(paramGradAcc.stride == 1)

          // paramGradAcc(dEw) += dEh * input.t * factor // Derivation by weights
          Dger.dger(outputSize, inputSize, paramFactor,
            dEh.data, dEh.offset, dEh.stride,
            input.data, input.offset, input.stride,
            paramGradAcc.data, paramGradAcc.offset, outputSize)

          // paramGradAcc(dEb) += dEh * factor // Derivation by biases
          Daxpy.daxpy(outputSize, paramFactor,
            dEh.data, dEh.offset, dEh.stride,
            paramGradAcc.data, paramGradAcc.offset + outputSize * inputSize, 1);
        }
      }

    }
  }

  def updateParams(gradient: DenseVector[Double]) =
    params += gradient

  def assignParams(newParams: DenseVector[Double]) =
    params := newParams

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

  protected def outputTransform(v: DenseVector[Double]) =
    sigmoid.inPlace(v)

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