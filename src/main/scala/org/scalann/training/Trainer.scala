package org.scalann.training

import org.scalann._
import org.scalann.decay.Decay
import breeze.linalg._

class Trainer(val learningRate: Double, val momentumMultiplier: Double, val decay: Decay, val decayCoeff: Double, val maxIter: Int) {

  def train[T](target: Optimizable[T])(examples: => Traversable[T])(callback: (Int) => Unit = { _ => }) {
    val momentum = DenseVector.zeros[Double](target.paramSize)

    for (iter <- 1 to maxIter) {
      momentum *= momentumMultiplier

      // Add weight decay
      decay.gradientAdd(target.params, target.paramsDecay)(momentum, learningRate * decayCoeff)

      // Add calculated gradient and update params
      target.gradientAdd(examples)(momentum, -learningRate)
      target.updateParams(momentum)

      callback(iter)
    }
  }

}
