package org.scalann

import breeze.linalg._
import com.google.caliper.SimpleBenchmark

class LearningBench extends SimpleBenchmark {

  val w = MnistReader.imagesReader.width
  val h = MnistReader.imagesReader.height

  val examples = (MnistReader.imagesReader.images zip MnistReader.labelsReader.labels).take(100).map {
    case (image, label) =>
      val input = DenseVector.tabulate(w * h) { i => image(i / w, i % w) / 255.0 }
      val output = DenseVector.zeros[Double](10)

      output(label) = 1.0

      input -> output
  }

  val example = examples.head

  val nn = new FeedForwardNetwork(List(new LogisticLayer(w * h, 200), new SoftmaxLayer(200, 10)))
  val l = new LogisticLayer(w * h, 10)

  override def setUp {
    examples.size
  }

  def timeLayerForward(reps: Int) {
    for (i <- 1 to reps)
      l.forward(example._1)
  }

  def timeLayerForwardBackward(reps: Int) {
    for (i <- 1 to reps)
      l.forward(example._1)._2.backward(example._2)
  }

  def timeNetworkForward(reps: Int) {
    for (i <- 1 to reps)
      nn.forward(example._1)
  }

  def timeNetworkForwardBackward(reps: Int) {
    for (i <- 1 to reps)
      nn.forward(example._1)._2.backward(example._2)
  }

}
