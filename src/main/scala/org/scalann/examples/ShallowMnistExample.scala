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
  val testExample = examples.drop(trainExamples.size).take(50)

  val nn = new FeedForwardNetwork(List(new BasicLayer(w * h, 200), new BasicLayer(200, 10)))

  val momentum = nn.examplesGradient(trainExamples)
  momentum *= 0 // TODO Implement zeroGradient

  for (iter <- 1 to 100) {
    val grad = nn.examplesGradient(trainExamples)
    grad *= -0.5

    momentum *= 0.7
    momentum += grad

    nn.update(momentum)

    println(nn.measureError(trainExamples))
  }
  
  for (iter <- 1 to 100) {    
    val test = trainExamples((math.random * 50).toInt)
    val out = nn(test._1)
    
    println(test._2, out)
    //readLine
  }

}