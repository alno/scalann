package org.scalann

import breeze.linalg._
import breeze.numerics._
import scala.math.exp
import org.netlib.blas.Dgemv
import org.netlib.blas.Daxpy
import org.netlib.blas.Dger
import org.scalann.activation.ActivationTransform

abstract class AbstractLayer(val inputSize: Int, val outputSize: Int) extends Stage {

  val paramSize = outputSize * (inputSize + 1)

  val params = DenseVector.fill(paramSize) { math.random * 2 - 1 }

  val weights: DenseMatrix[Double] = new DenseMatrix(outputSize, inputSize, params.data)
  val biases: DenseVector[Double] = new DenseVector(params.data, outputSize * inputSize, 1, outputSize)

  val paramsDecay =
    DenseVector.vertcat(DenseVector.fill(outputSize * inputSize)(1.0), DenseVector.fill(outputSize)(0.0))

  def forward(input: DenseVector[Double]) = {
    val result = weights * input
    result += biases
    activation.transformOutput(result)

    result -> new Memo {

      override def backwardAdd(derivation: DenseVector[Double], outputDeriv: Boolean)(inputGradAcc: DenseVector[Double], inputFactor: Double, paramGradAcc: DenseVector[Double], paramFactor: Double) {
        var dEh = derivation // Derivation by internal activations

        if (outputDeriv) { // If output derivation should be transformed
          dEh = dEh.copy // Copy it
          activation.transformOutputDerivation(dEh, result) // Transform (destructively)
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

  def activation: ActivationTransform

}
