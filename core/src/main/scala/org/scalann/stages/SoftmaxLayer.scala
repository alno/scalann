package org.scalann.stages

import breeze.linalg._
import org.scalann.loss.SoftmaxLoss
import org.scalann.activation.Softmax

class SoftmaxLayer(inputSize: Int, outputSize: Int) extends AbstractLayer(inputSize, outputSize) {

  override def activation = Softmax
  override def loss = SoftmaxLoss

  override def toString = s"Softmax($outputSize)"

}
