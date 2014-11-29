package org.scalann.activation

import breeze.linalg.DenseVector

trait ActivationTransform {

  def transformOutput(v: DenseVector[Double]): Unit

  def transformOutputDerivation(dv: DenseVector[Double], v: DenseVector[Double]): Unit

}
