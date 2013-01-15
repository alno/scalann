package org.scalann.examples

import org.scalann._
import breeze.linalg._
import java.io.DataOutputStream
import java.io.FileOutputStream
import org.scalann.training.NetworkTrainer

trait TrainingAlg {

  import Utils._

  def learningRate: Double
  def weightDecay: Double
  def momentumMult: Double

}

object ShallowMnistExample extends App with TrainingAlg {

  import Utils._

  val w = MnistReader.imagesReader.width
  val h = MnistReader.imagesReader.height

  val examples = (MnistReader.imagesReader.images zip MnistReader.labelsReader.labels).map {
    case (image, label) =>
      val input = DenseVector.tabulate(w * h) { i => image(i / w, i % w) / 255.0 }
      val output = DenseVector.zeros[Double](10)

      output(label) = 1.0

      input -> output
  }

  val trainExamples = examples.take(7000).toList
  val testExamples = examples.drop(trainExamples.size).take(10000).toList

  val learningRate = 0.5
  val weightDecay = 0.005
  val momentumMult = 0.8

  val nn = new FeedForwardNetwork(List(new LogisticLayer(w * h, 50), new SoftmaxLayer(50, 10)))

  new NetworkTrainer(learningRate, weightDecay, momentumMult).train(nn)(trainExamples)

  println("Training loss: " + nn.examplesLoss(trainExamples))
  println("Test loss: " + nn.examplesLoss(testExamples))

  val testErrorRate = testExamples.filter { ex =>
    nn(ex._1).activeIterator.maxBy(_._2)._1 != ex._2.activeIterator.maxBy(_._2)._1
  }.size * 1.0 / testExamples.size

  println("Test error rate: " + testErrorRate)

  nn.save(new DataOutputStream(new FileOutputStream("/home/alno/nn-simple-wd.dat")))

  println("Params saved")

}