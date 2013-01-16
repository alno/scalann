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

  val trainExamples = MnistReader.examples.take(7000).toList
  val testExamples = MnistReader.examples.drop(trainExamples.size).take(10000).toList

  val learningRate = 0.5
  val weightDecay = 0.005
  val momentumMult = 0.8

  val nn = new FeedForwardNetwork(List(new LogisticLayer(w * h, 50), new SoftmaxLayer(50, 10)))

  new NetworkTrainer(learningRate, weightDecay, momentumMult).train(nn)(trainExamples)

  println("Training loss: " + nn.examplesLoss(trainExamples))
  println("Test loss: " + nn.examplesLoss(testExamples))

  val testErrorRate = testExamples.filter { ex =>
    nn(ex._1).argmax != ex._2.argmax
  }.size * 1.0 / testExamples.size

  println("Test error rate: " + testErrorRate)

  nn.save(new DataOutputStream(new FileOutputStream("/home/alno/nn-simple-wd.dat")))

  println("Params saved")

}