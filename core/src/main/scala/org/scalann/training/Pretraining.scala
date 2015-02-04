package org.scalann.training

import org.scalann._
import org.scalann.utils._
import org.scalann.decay.Decay
import breeze.linalg._

trait Pretraining extends Trainer {

  def createRbmTrainer(layer: LogisticLayer, rbm: Rbm): Trainer =
    new Trainer(learningRate, momentumMultiplier, decay, maxIter)

  def createOutPreTrainer(layer: Stage): Trainer =
    new Trainer(learningRate, momentumMultiplier, decay, maxIter)

  override abstract def train[T](target: Optimizable[T])(examples: => IndexedSeq[target.Example])(callback: (Int) => Unit = { _ => }) {
    target match {
      case nn: SequentalNetwork =>
        pretrain(nn)(examples)
      case _ =>
    }

    super.train(target)(examples)(callback)
  }

  def pretrain(nn: SequentalNetwork)(examples: => IndexedSeq[(DenseVector[Double], DenseVector[Double])]) {
    var currentPretrainInputs = examples.map(_._1)

    for (l <- 0 until nn.layers.size - 1) {
      println("Pretraining layer " + l)

      val layer = nn.layers(l).asInstanceOf[LogisticLayer]
      val rbm = new Rbm(layer.inputSize, layer.outputSize)

      // Train rbm
      createRbmTrainer(layer, rbm).train(rbm)(currentPretrainInputs.sample(50))() // TODO Notify

      // Assign pretrained layer weights and biases
      layer.weights := rbm.weights
      layer.biases := rbm.hiddenBiases

      println("Processing pretrain inputs...")
      currentPretrainInputs = currentPretrainInputs.map(layer)
    }

    val lastPretrainExamples = currentPretrainInputs zip examples.view.map(_._2)
    val lastLayer = nn.layers.last

    println("Pretraining output layer")
    createOutPreTrainer(lastLayer).train(lastLayer)(lastPretrainExamples.sample(500))() // TODO Notify
    println("Pretraining complete")

    // TODO Pretrain up in a loop
  }

}
