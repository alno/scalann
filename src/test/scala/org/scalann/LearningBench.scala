package org.scalann

import breeze.linalg._
import com.google.caliper.SimpleBenchmark
import org.scalann.examples.Mnist

class LearningBench extends SimpleBenchmark {

  val mnist = new Mnist(System.getenv("MNIST_PATH"))
  val examples = mnist.examples

  val example = examples.head
  val multiExamples = examples.take(100).toList

  val input = example._1
  val multiInputs = examples.map(_._1)

  val nn = new SequentalNetwork(List(new LogisticLayer(mnist.imageWidth * mnist.imageHeight, 200), new SoftmaxLayer(200, 10)))
  val l = new LogisticLayer(mnist.imageWidth * mnist.imageHeight, 10)
  val rbm = new Rbm(mnist.imageWidth * mnist.imageHeight, 10)

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

  def timeLayerMultipleExamplesLoss(reps: Int) {
    for (i <- 1 to reps)
      nn.examplesLoss(multiExamples)
  }

  def timeLayerMultipleExamplesGradient(reps: Int) {
    for (i <- 1 to reps)
      nn.gradient(multiExamples)
  }

  def timeNetworkForward(reps: Int) {
    for (i <- 1 to reps)
      nn.forward(example._1)
  }

  def timeNetworkForwardBackward(reps: Int) {
    for (i <- 1 to reps)
      nn.forward(example._1)._2.backward(example._2)
  }

  def timeNetworkMultipleExamplesLoss(reps: Int) {
    for (i <- 1 to reps)
      nn.examplesLoss(multiExamples)
  }

  def timeNetworkMultipleExamplesGradient(reps: Int) {
    for (i <- 1 to reps)
      nn.gradient(multiExamples)
  }

  def timeRbmGradient(reps: Int) {
    for (i <- 1 to reps)
      rbm.gradient(input)
  }

  def timeRbmMultipleExamplesGradient(reps: Int) {
    for (i <- 1 to reps)
      rbm.gradient(multiInputs)
  }

}
