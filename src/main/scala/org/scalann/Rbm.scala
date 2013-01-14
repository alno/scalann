package org.scalann

import breeze.linalg._
import breeze.numerics._
import Utils._
import org.netlib.blas.Dgemm

class Rbm(val inputSize: Int, val outputSize: Int) extends Optimizable[DenseVector[Double]] {
  val paramSize = outputSize * inputSize

  val params = DenseVector.fill(paramSize) { math.random * 2 - 1 }
  val weights: DenseMatrix[Double] = new DenseMatrix(outputSize, inputSize, params.data)

  private val weightsT: DenseMatrix[Double] = weights.t

  def updateParams(gradient: DenseVector[Double]) =
    params += gradient

  def assignParams(newParams: DenseVector[Double]) =
    params := newParams

  def gradientAdd(example: DenseVector[Double])(paramGradAcc: DenseVector[Double], factor: Double) {
    if (paramGradAcc == null)
      return

    val visibleData0 = sample(example)

    val hiddenData0 = weights * visibleData0
    sigmoid.inPlace(hiddenData0)
    sample.inPlace(hiddenData0)

    val visibleData1 = weightsT * hiddenData0
    sigmoid.inPlace(visibleData1)
    sample.inPlace(visibleData1)

    val hiddenProb1 = weights * visibleData1
    sigmoid.inPlace(hiddenProb1)

    // paramGradAcc += hiddenData0 * visibleData0.t * factor
    Dgemm.dgemm("n", "t", outputSize, inputSize, 1,
      factor, hiddenData0.data, hiddenData0.offset, outputSize,
      visibleData0.data, visibleData0.offset, inputSize,
      1.0, paramGradAcc.data, paramGradAcc.offset, outputSize)

    // paramGradAcc -= hiddenProb1 * visibleData1.t * factor
    Dgemm.dgemm("n", "t", outputSize, inputSize, 1,
      -factor, hiddenProb1.data, hiddenProb1.offset, outputSize,
      visibleData1.data, visibleData1.offset, inputSize,
      1.0, paramGradAcc.data, paramGradAcc.offset, outputSize)
  }

}