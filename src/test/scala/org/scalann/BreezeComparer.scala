package org.scalann

import breeze.linalg._
import breeze.classify._
import breeze.data.Example
import breeze.optimize.FirstOrderMinimizer.OptParams

object BreezeComparer extends App {

  val w = MnistReader.imagesReader.width
  val h = MnistReader.imagesReader.height

  val examples = MnistReader.imagesReader.images zip MnistReader.labelsReader.labels

  val matrixExamples = examples.map {
    case (image, label) =>
      val input = DenseVector.tabulate(w * h) { i => image(i / w, i % w) / 255.0 }
      val output = DenseVector.zeros[Double](10)

      output(label) = 1.0

      input -> output
  }

  val breezeExamples = examples.map {
    case (image, label) =>
      val input = Array.tabulate(w * h) { i => i -> image(i / w, i % w) / 255.0 }

      Example(label, Counter(input))
  }

  val breezeTrainExamples = breezeExamples.take(10).toList
  val breezeTestExamples = breezeExamples.drop(10).take(1000).toList

  val params = OptParams(maxIterations = 10)
  val trainer = new NNetClassifier.CounterTrainer[Int, Int](params)
  
  println("Breeze training started")
  val nnet = trainer.train(breezeExamples)
  println("Breeze training completed")

  val breezeErrorRate = breezeTestExamples.view.filter { ex =>
    nnet(ex.features) != ex.label
  }.size

  println("Breeze error rate: " + breezeErrorRate)
}