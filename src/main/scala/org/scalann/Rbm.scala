package org.scalann

import breeze.linalg._
import breeze.numerics._
import Utils._
import org.netlib.blas.Dgemm
import org.netlib.blas.Dgemv

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

    val visible = example.copy
    val hidden = DenseVector.zeros[Double](outputSize)

    sample.inPlace(visible)

    // hidden = weights * visible
    Dgemv.dgemv("n", outputSize, inputSize,
      1.0, weights.data, weights.offset, weights.majorStride,
      visible.data, visible.offset, 1,
      0.0, hidden.data, hidden.offset, 1)

    sigmoid.inPlace(hidden)
    sample.inPlace(hidden)

    // paramGradAcc += hidden * visible.t * factor
    Dgemm.dgemm("n", "t", outputSize, inputSize, 1,
      factor, hidden.data, hidden.offset, outputSize,
      visible.data, visible.offset, inputSize,
      1.0, paramGradAcc.data, paramGradAcc.offset, outputSize)

    // visible = weights.t * hidden
    Dgemv.dgemv("t", outputSize, inputSize,
      1.0, weights.data, weights.offset, weights.majorStride,
      hidden.data, hidden.offset, 1,
      0.0, visible.data, visible.offset, 1)

    sigmoid.inPlace(visible)
    sample.inPlace(visible)

    // hidden = weights * visible
    Dgemv.dgemv("n", outputSize, inputSize,
      1.0, weights.data, weights.offset, weights.majorStride,
      visible.data, visible.offset, 1,
      0.0, hidden.data, hidden.offset, 1)

    sigmoid.inPlace(hidden)
    sample.inPlace(hidden)

    // paramGradAcc -= hiddenProb1 * visibleData1.t * factor
    Dgemm.dgemm("n", "t", outputSize, inputSize, 1,
      -factor, hidden.data, hidden.offset, outputSize,
      visible.data, visible.offset, inputSize,
      1.0, paramGradAcc.data, paramGradAcc.offset, outputSize)
  }

}