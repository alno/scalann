package org.scalann.examples

import org.scalann._
import breeze.linalg._

object ShallowMnistExample extends App {

  val w = MnistReader.imagesReader.width
  val h = MnistReader.imagesReader.height

  val examples = (MnistReader.imagesReader.images zip MnistReader.labelsReader.labels).map {
    case (image, label) =>
      val input = DenseVector.tabulate(w * h) { i => image(i / w, i % w) / 255.0 }
      val output = DenseVector.zeros[Double](10)

      output(label) = 1.0

      input -> output
  }

  val trainExamples = examples.take(50)
  val testExamples = examples.drop(trainExamples.size).take(5000)

  val nn = new FeedForwardNetwork(List(new LogisticLayer(w * h, 200), new SoftmaxLayer(200, 10)))

  val momentum = nn.examplesGradient(trainExamples)
  momentum *= 0 // TODO Implement zeroGradient

  for (iter <- 1 to 100) {
    val grad = nn.examplesGradient(trainExamples)
    grad *= -0.5

    momentum *= 0.7
    momentum += grad

    nn.update(momentum)

    println(nn.examplesLoss(trainExamples))
  }

  println("Training loss: " + nn.examplesLoss(trainExamples))
  println("Test loss: " + nn.examplesLoss(testExamples))

  val testErrorRate = testExamples.filter { ex =>
    nn(ex._1).activeIterator.maxBy(_._2)._1 != ex._2.activeIterator.maxBy(_._2)._1
  }.size * 1.0 / testExamples.size

  println("Test error rate: " + testErrorRate)

}