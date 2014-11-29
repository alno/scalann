package org.scalann

import breeze.linalg._
import org.scalann.loss.SquaredLoss

class LinearLayer(inputSize: Int, outputSize: Int) extends AbstractLayer(inputSize, outputSize) {

  protected def outputTransform(v: DenseVector[Double]) {}

  protected def outputDerivationTransform(dv: DenseVector[Double], v: DenseVector[Double]) {}

  override def loss = SquaredLoss

}
