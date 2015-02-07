package org.scalann.stages

import breeze.linalg._
import breeze.numerics._
import org.scalann.loss.LogisticLoss
import org.scalann.activation.Logistic

class LogisticLayer(inputSize: Int, outputSize: Int) extends AbstractLayer(inputSize, outputSize) {

  override def activation = Logistic
  override def loss = LogisticLoss

  override def toString = s"Logistic($outputSize)"

}
