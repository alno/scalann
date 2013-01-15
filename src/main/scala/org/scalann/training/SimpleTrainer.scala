package org.scalann.training

import org.scalann._
import breeze.linalg._

class SimpleTrainer(val learningRate: Double, val weightDecay: Double, val momentumMultiplier: Double) {

  def train(stage: Stage)(examples: List[(DenseVector[Double], DenseVector[Double])]) {
    val momentum = DenseVector.zeros[Double](stage.paramSize)

    for (iter <- 1 to 200) {
      val wd = stage.paramsDecay :* stage.params
      wd *= -weightDecay

      if (iter % 5 == 0) {
        val exLoss = stage.examplesLoss(examples)
        val wdLoss = 0.5 * weightDecay * wd.dot(wd)

        println(iter + ": " + (exLoss + wdLoss) + ", " + exLoss + " + " + wdLoss)
      }

      momentum *= momentumMultiplier
      momentum += wd

      stage.gradientAdd(examples)(momentum, -learningRate) // Add gradient
      stage.updateParams(momentum)
    }
  }

}