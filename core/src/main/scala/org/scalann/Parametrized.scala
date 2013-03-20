package org.scalann

import breeze.linalg._

trait Parametrized {

  def paramSize: Int
  def params: DenseVector[Double]

  /**
   * Coefficients for param decay - used to turn off decay of some parameters (such as biases)
   */
  def paramsDecay: DenseVector[Double]

  def assignParams(newParams: DenseVector[Double])
  def updateParams(addParams: DenseVector[Double])

}