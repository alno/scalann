package org.scalann.examples

import breeze.linalg._
import breeze.optimize.FirstOrderMinimizer.OptParams
import nak.classify._
import nak.data.Example

object BreezeComparer extends App {

  val mnist = Mnist.trainDataset

  val breezeExamples = (mnist.imagesAsMatrices zip mnist.labelsAsInts).map {
    case (image, label) =>
      val input = Array.tabulate(mnist.imageWidth * mnist.imageHeight) { i => i -> image(i / mnist.imageWidth, i % mnist.imageWidth) / 255.0 }

      Example(label, Counter(input))
  }

  val breezeTrainExamples = breezeExamples.take(10).toList
  val breezeTestExamples = breezeExamples.drop(10).take(100).toList

  val trainer = new NNetClassifier.CounterTrainer[Int, Int](OptParams(maxIterations = 10))

  println("Breeze training started")
  val nnet = trainer.train(breezeExamples)
  println("Breeze training completed")

  val breezeErrorRate = breezeTestExamples.view.filter { ex =>
    nnet(ex.features) != ex.label
  }.size

  println("Breeze error rate: " + breezeErrorRate)
}