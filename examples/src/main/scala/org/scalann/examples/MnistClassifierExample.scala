package org.scalann.examples

import org.scalann._
import org.scalann.utils._
import org.scalann.decay._
import org.scalann.training._
import org.scalann.visualization._
import java.io.{ DataOutputStream, FileOutputStream }
import breeze.linalg._

object MnistClassifierExample extends App {

  val mnist = new Mnist(System.getenv("MNIST_PATH"))

  val trainExamples = mnist.examples.take(7000).toVector
  val testExamples = mnist.examples.drop(trainExamples.size).take(10000).toVector

  val trainer = new NetworkTrainer(
    learningRate = 0.5,
    momentumMultiplier = 0.8,
    decay = L2Decay,
    decayCoeff = 0.005,
    maxIter = 2000)

  val nn = new SequentalNetwork(List(new LogisticLayer(mnist.imageWidth * mnist.imageHeight, 25), new SoftmaxLayer(25, 10)))

  trainer.train(nn) { trainExamples } { iter =>
    if (iter % 10 == 0) {
      println(iter)
    }

    if (iter % 200 == 0) {
      println("Training loss: " + nn.examplesLoss(trainExamples))

      ImageUtils.saveLayerWeight(nn.layers(0).asInstanceOf[LogisticLayer].weights, "mnist-nn.png", mnist.imageWidth, mnist.imageHeight, 5, 5)
      println("First layer image saved")

      nn.save(new DataOutputStream(new FileOutputStream("mnist-nn.dat")))
      println("Parameters saved")
    }
  }

  println("Training loss: " + nn.examplesLoss(trainExamples))
  println("Test loss: " + nn.examplesLoss(testExamples))

  val testErrorRate = testExamples.filter { ex =>
    argmax(nn(ex._1)) != argmax(ex._2)
  }.size * 1.0 / testExamples.size

  println("Finished, test error rate: " + testErrorRate)

}