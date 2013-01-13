package org.scalann

import breeze.linalg._
import breeze.numerics._
import Utils._

class Rbm(val inputSize: Int, val outputSize: Int) extends Parametrized {
  val paramSize = outputSize * inputSize

  val params = DenseVector.fill(paramSize) { math.random * 2 - 1 }
  val weights: DenseMatrix[Double] = new DenseMatrix(outputSize, inputSize, params.data)
 
  private val weightsT: DenseMatrix[Double] = weights.t

  private def visibleStateToHiddenProbs(visible: DenseVector[Double]): DenseVector[Double] = {
    val result = weights * visible
    sigmoid.inPlace(result)
    result
  }

  private def hiddenStateToVisibleProbs(hidden: DenseVector[Double]): DenseVector[Double] = {
    val result = weightsT * hidden
    sigmoid.inPlace(result)
    result
  }
  
  def updateParams(gradient: DenseVector[Double]) =
    params += gradient

  def assignParams(newParams: DenseVector[Double]) =
    params := newParams
  
  def gradient(visibleProb0: DenseVector[Double]): DenseVector[Double] = {
    val res = DenseVector.zeros[Double](paramSize)
    
    val visibleData0 = sample(visibleProb0)

    val hiddenProb0 = visibleStateToHiddenProbs(visibleData0)
    val hiddenData0 = sample(hiddenProb0)
 
    val visibleProb1 = hiddenStateToVisibleProbs(hiddenData0)
    val visibleData1 = sample(visibleProb1)
       
    val hiddenProb1 = visibleStateToHiddenProbs(visibleData1)
    
    new DenseMatrix(outputSize, inputSize, res.data) := hiddenData0 * visibleData0.t - hiddenProb1 * visibleData1.t
    res
  }
  
  def gradient(examples: Traversable[DenseVector[Double]]): DenseVector[Double] = {
    val grad = gradient(examples.head)
    
    examples.tail.foreach { ex =>
      grad += gradient(ex)
    }
    
    grad *= 1.0 / examples.size
    grad
  }
  
}