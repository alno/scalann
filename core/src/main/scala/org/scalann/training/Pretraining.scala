package org.scalann.training

import org.scalann._
import org.scalann.utils._
import org.scalann.decay.Decay
import breeze.linalg._
import org.scalann.stages.SeqStage
import org.scalann.stages.LogisticLayer

trait Pretraining extends Trainer {

  def createRbmTrainer(layer: LogisticLayer, rbm: Rbm): Trainer =
    new Trainer(learningRate, momentumMultiplier, decay, maxIter)

  def createOutPreTrainer(layer: Stage): Trainer =
    new Trainer(learningRate, momentumMultiplier, decay, maxIter)

  override abstract def train[T](target: Optimizable[T])(examples: => IndexedSeq[target.Example])(callback: (Int) => Unit = { _ => }) {
    target match {
      case nn: SeqStage[_, _] =>
        pretrain(nn)(examples)
      case _ =>
    }

    super.train(target)(examples)(callback)
  }

  def pretrain(nn: SeqStage[_, _])(examples: IndexedSeq[(DenseVector[Double], DenseVector[Double])]) {
    val inputs = examples.map(_._1)
    val layer = nn.head.asInstanceOf[LogisticLayer]

    val rbm = new Rbm(layer.inputSize, layer.outputSize)

    // Train rbm
    createRbmTrainer(layer, rbm).train(rbm)(inputs.sample(50))() // TODO Notify

    // Assign pretrained layer weights and biases
    layer.weights := rbm.weights
    layer.biases := rbm.hiddenBiases

    val nextExamples = inputs.map(layer) zip examples.map(_._2)

    nn.tail match {
      case nnn: SeqStage[_, _] =>
        pretrain(nnn)(nextExamples)
      case outputLayer: Stage =>
        println("Pretraining output layer")
        createOutPreTrainer(outputLayer).train(outputLayer)(nextExamples)() // TODO Notify
        println("Pretraining complete")
    }
  }

}
