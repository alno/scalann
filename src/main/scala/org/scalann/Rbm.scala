package org.scalann

import breeze.linalg._
import breeze.numerics._
import Utils._
import org.netlib.blas.Dgemm
import org.netlib.blas.Dgemv
import org.netlib.blas.Daxpy

class Rbm(val visibleSize: Int, val hiddenSize: Int) extends Optimizable[DenseVector[Double]] {
  val paramSize = hiddenSize * visibleSize + visibleSize + hiddenSize

  val params = DenseVector.fill(paramSize) { math.random * 2 - 1 }
  val weights: DenseMatrix[Double] = new DenseMatrix(hiddenSize, visibleSize, params.data)
  val visibleBiases: DenseVector[Double] = new DenseVector(params.data, hiddenSize * visibleSize, 1, visibleSize)
  val hiddenBiases: DenseVector[Double] = new DenseVector(params.data, hiddenSize * visibleSize + visibleSize, 1, hiddenSize)

  val paramsDecay =
    DenseVector.vertcat(DenseVector.fill(hiddenSize * visibleSize)(1.0), DenseVector.fill(hiddenSize + visibleSize)(0.0))

  var cdLevel = 2

  def updateParams(gradient: DenseVector[Double]) =
    params += gradient

  def assignParams(newParams: DenseVector[Double]) =
    params := newParams

  def reconstruction(hidden: DenseVector[Double]) =
    weights.t * sample(hidden) + visibleBiases

  def reconstruction(level: Int): DenseVector[Double] = {
    val visible = DenseVector.zeros[Double](visibleSize)
    val hidden = DenseVector.fill[Double](hiddenSize)(0.5)

    for (i <- 1 to level)
      updateHiddenProb(hidden, visible)

    reconstruction(hidden)
  }

  /**
   * Calculating gradient with CDn algorithm
   * Tuning according to: http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
   */
  def gradientAdd(example: DenseVector[Double])(paramGradAcc: DenseVector[Double], factor: Double) {
    if (paramGradAcc == null)
      return

    val visible = example.copy
    val hidden = DenseVector.zeros[Double](hiddenSize)

    // hidden = weights * visible
    Dgemv.dgemv("n", hiddenSize, visibleSize,
      1.0, weights.data, weights.offset, weights.majorStride,
      visible.data, visible.offset, 1,
      0.0, hidden.data, hidden.offset, 1)

    hidden += hiddenBiases
    sigmoid.inPlace(hidden)

    // Updating gradient with positive statistics

    // paramGradAcc(weights) += hidden * visible.t * factor
    Dgemm.dgemm("n", "t", hiddenSize, visibleSize, 1,
      factor, hidden.data, hidden.offset, hiddenSize,
      visible.data, visible.offset, visibleSize,
      1.0, paramGradAcc.data, paramGradAcc.offset, hiddenSize)

    // paramGradAcc(visibleBiases) += visible * factor
    Daxpy.daxpy(visibleSize, factor,
      visible.data, visible.offset, 1,
      paramGradAcc.data, paramGradAcc.offset + hiddenSize * visibleSize, 1);

    // paramGradAcc(hiddenBiases) += hidden * factor
    Daxpy.daxpy(hiddenSize, factor,
      hidden.data, hidden.offset, 1,
      paramGradAcc.data, paramGradAcc.offset + hiddenSize * visibleSize + visibleSize, 1);

    // Iterating cdLevel times to generate reconstructions
    for (i <- 1 to cdLevel)
      updateHiddenProb(hidden, visible)

    // Updating gradient with negative statistics

    // paramGradAcc -= hiddenProb1 * visibleData1.t * factor
    Dgemm.dgemm("n", "t", hiddenSize, visibleSize, 1,
      -factor, hidden.data, hidden.offset, hiddenSize,
      visible.data, visible.offset, visibleSize,
      1.0, paramGradAcc.data, paramGradAcc.offset, hiddenSize)

    // paramGradAcc(visibleBiases) -= visible * factor
    Daxpy.daxpy(visibleSize, -factor,
      visible.data, visible.offset, 1,
      paramGradAcc.data, paramGradAcc.offset + hiddenSize * visibleSize, 1);

    // paramGradAcc(hiddenBiases) -= hidden * factor
    Daxpy.daxpy(hiddenSize, -factor,
      hidden.data, hidden.offset, 1,
      paramGradAcc.data, paramGradAcc.offset + hiddenSize * visibleSize + visibleSize, 1);
  }

  private def updateHiddenProb(hidden: DenseVector[Double], visible: DenseVector[Double]) {
    // Sample hidden state to create information bottleneck
    sample.inPlace(hidden)

    // visible = weights.t * hidden
    Dgemv.dgemv("t", hiddenSize, visibleSize,
      1.0, weights.data, weights.offset, weights.majorStride,
      hidden.data, hidden.offset, 1,
      0.0, visible.data, visible.offset, 1)

    visible += visibleBiases
    sigmoid.inPlace(visible)

    // hidden = weights * visible
    Dgemv.dgemv("n", hiddenSize, visibleSize,
      1.0, weights.data, weights.offset, weights.majorStride,
      visible.data, visible.offset, 1,
      0.0, hidden.data, hidden.offset, 1)

    hidden += hiddenBiases
    sigmoid.inPlace(hidden)
  }

}