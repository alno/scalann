package org.scalann.examples

import org.scalann._
import org.scalann.utils._
import org.scalann.decay._
import org.scalann.training._
import org.scalann.visualization._
import java.io.{ DataOutputStream, FileOutputStream }

object ShallowRbmExample extends App {

  require(args.size == 1, "should have MNIST location as argument")

  val mnist = new Mnist(args(0))

  val trainImages = mnist.imagesAsVectors.take(1000).toVector

  val trainer = new SimpleTrainer(
    learningRate = 0.2,
    momentumMultiplier = 0.5,
    decay = L2Decay,
    decayCoeff = 0.001,
    maxIter = 10000)

  val rbm = new Rbm(mnist.imageWidth * mnist.imageHeight, 25)

  trainer.train(rbm) { trainImages.sample(30) } { iter =>
    if (iter % 10 == 0) {
      println(iter)
    }

    if (iter % 200 == 0) {
      ImageUtils.saveLayerWeight(rbm.weights, "mnist-rbm.png", mnist.imageWidth, mnist.imageHeight, 5, 5)
      println("Image saved")

      rbm.save(new DataOutputStream(new FileOutputStream("mnist-rbm.dat")))
      println("Parameters saved")
    }
  }

  println("Finished")

}
