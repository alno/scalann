package org.scalann.training

import org.scalann._
import breeze.linalg._

class RbmTrainer(val learningRate: Double, val weightDecay: Double, val momentumMultiplier: Double) {

  def train(rbm: Rbm)(examples: List[DenseVector[Double]]) {
    val momentum = DenseVector.zeros[Double](rbm.paramSize)

    // Pretrain layer
    for (iter <- 1 to 100) {
      println(iter)

      momentum *= momentumMultiplier
      rbm.gradientAdd(examples)(momentum, learningRate)
      rbm.updateParams(momentum)
    }
  }

}