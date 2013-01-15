package org.scalann.training

import org.scalann._
import breeze.linalg._

class NetworkTrainer(val learningRate: Double, val weightDecay: Double, val momentumMultiplier: Double) {

  private val stageTrainer = new SimpleTrainer(learningRate, weightDecay, momentumMultiplier)
  private val rbmTrainer = new RbmTrainer(learningRate, weightDecay, momentumMultiplier)

  def train(nn: FeedForwardNetwork)(examples: List[(DenseVector[Double], DenseVector[Double])]) {
    pretrainNetwork(nn)(examples)
    stageTrainer.train(nn)(examples)
  }

  def pretrainNetwork(nn: FeedForwardNetwork)(examples: List[(DenseVector[Double], DenseVector[Double])]) {
    var currentPretrainInputs = examples.map(_._1)

    for (l <- 0 until nn.layers.size - 1) {
      println("Pretraining layer " + l)

      val layer = nn.layers(l).asInstanceOf[LogisticLayer]
      val rbm = new Rbm(layer.inputSize, layer.outputSize)

      // Train rbm
      rbmTrainer.train(rbm)(currentPretrainInputs)

      // Assign pretrained layer weights
      layer.weights := rbm.weights

      println("Processing pretrain inputs...")
      currentPretrainInputs = currentPretrainInputs.map(layer)
    }

    val lastPretrainExamples = currentPretrainInputs zip examples.view.map(_._2)
    val lastLayer = nn.layers.last

    println("Pretraining output layer")
    stageTrainer.train(lastLayer)(lastPretrainExamples)
    println("Pretraining complete")
  }

}