package org.scalann.training

import org.scalann._
import org.scalann.decay.Decay
import breeze.linalg._

class NetworkTrainer(val learningRate: Double, val momentumMultiplier: Double, val decay: Decay, val decayCoeff: Double, val maxIter: Int) {

  private val stageTrainer = new SimpleTrainer(learningRate, momentumMultiplier, decay, decayCoeff, maxIter)
  private val rbmTrainer = new SimpleTrainer(learningRate, momentumMultiplier, decay, decayCoeff, maxIter)

  def train(nn: SequentalNetwork)(examples: Seq[(DenseVector[Double], DenseVector[Double])])(callback: (Int) => Unit = { _ => }) {
    pretrainNetwork(nn)(examples)
    stageTrainer.train(nn)(examples)(callback)
  }

  def pretrainNetwork(nn: SequentalNetwork)(examples: Seq[(DenseVector[Double], DenseVector[Double])]) {
    var currentPretrainInputs = examples.map(_._1)

    for (l <- 0 until nn.layers.size - 1) {
      println("Pretraining layer " + l)

      val layer = nn.layers(l).asInstanceOf[LogisticLayer]
      val rbm = new Rbm(layer.inputSize, layer.outputSize)

      // Train rbm
      rbmTrainer.train(rbm)(currentPretrainInputs)()

      // Assign pretrained layer weights
      layer.weights := rbm.weights

      println("Processing pretrain inputs...")
      currentPretrainInputs = currentPretrainInputs.map(layer)
    }

    val lastPretrainExamples = currentPretrainInputs zip examples.view.map(_._2)
    val lastLayer = nn.layers.last

    println("Pretraining output layer")
    stageTrainer.train(lastLayer)(lastPretrainExamples)()
    println("Pretraining complete")
  }

}
