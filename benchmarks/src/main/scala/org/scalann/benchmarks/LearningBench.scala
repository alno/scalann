package org.scalann.examples

import breeze.linalg._
import org.scalann.LogisticLayer
import org.scalann.Rbm
import org.scalann.SequentalNetwork
import org.scalann.SoftmaxLayer
import org.openjdk.jmh.annotations._

@State(Scope.Thread)
class LearningBench {
  import LearningBench._

  val nn = new SequentalNetwork(List(new LogisticLayer(mnist.imageWidth * mnist.imageHeight, 200), new SoftmaxLayer(200, 10)))
  val l = new LogisticLayer(mnist.imageWidth * mnist.imageHeight, 10)
  val rbm = new Rbm(mnist.imageWidth * mnist.imageHeight, 10)

  @Benchmark
  def timeLayerForward {
    l.forward(example._1)
  }

  @Benchmark
  def timeLayerForwardBackward {
    l.forward(example._1)._2.backward(example._2)
  }

  @Benchmark
  def timeLayerMultipleExamplesLoss {
    nn.examplesLoss(multiExamples)
  }

  @Benchmark
  def timeLayerMultipleExamplesGradient {
    nn.gradient(multiExamples)
  }

  @Benchmark
  def timeNetworkForward {
    nn.forward(example._1)
  }

  @Benchmark
  def timeNetworkForwardBackward {
    nn.forward(example._1)._2.backward(example._2)
  }

  @Benchmark
  def timeNetworkMultipleExamplesLoss {
    nn.examplesLoss(multiExamples)
  }

  @Benchmark
  def timeNetworkMultipleExamplesGradient {
    nn.gradient(multiExamples)
  }

  @Benchmark
  def timeRbmGradient {
    rbm.gradient(input)
  }

  @Benchmark
  def timeRbmMultipleExamplesGradient {
    rbm.gradient(multiInputs)
  }

}

object LearningBench {
  val mnist = Mnist.trainDataset
  val examples = mnist.examples

  val example = examples.head
  val multiExamples = examples.take(100).toVector

  val input = example._1
  val multiInputs = examples.map(_._1)
}