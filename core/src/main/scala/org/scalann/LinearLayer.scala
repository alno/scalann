package org.scalann

import breeze.linalg._
import org.scalann.loss.SquaredLoss
import org.scalann.activation.Linear

class LinearLayer(inputSize: Int, outputSize: Int) extends AbstractLayer(inputSize, outputSize) {

  override def activation = Linear
  override def loss = SquaredLoss

  override def toString = s"linear($outputSize)"

}
