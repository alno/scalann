package org.scalann.activation

import breeze.linalg._
import breeze.numerics._

object Linear extends ActivationTransform {

  override def transformOutput(v: DenseVector[Double]): Unit = ()

  override def transformOutputDerivation(dv: DenseVector[Double], v: DenseVector[Double]): Unit = ()

}
