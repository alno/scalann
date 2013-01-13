package org.scalann

import breeze.linalg._

trait Parametrized {

  def paramSize: Int
  def params: DenseVector[Double]

  def assignParams(newParams: DenseVector[Double])
  def updateParams(addParams: DenseVector[Double])

}