package org.scalann.examples

import org.scalann._
import breeze.linalg._
import java.io.DataOutputStream
import java.io.FileOutputStream

object ShallowRbmExample extends App {

  val w = MnistReader.imagesReader.width
  val h = MnistReader.imagesReader.height

  val images = MnistReader.imagesReader.images.map { image =>
    DenseVector.tabulate(w * h) { i => image(i / w, i % w) / 255.0 }
  }

  val trainImages = images.take(5000)

  val learningRate = 0.5
  val weightDecay = 0.05
  val momentumMult = 0.3

  val rbm = new Rbm(w * h, 50)

  val momentum = DenseVector.zeros[Double](rbm.paramSize)

  for (iter <- 1 to 5000) {
    println(iter)

    val grad = rbm.gradient(trainImages)
    grad *= learningRate

    momentum *= momentumMult
    momentum += grad

    rbm.updateParams(momentum)
  }

  println("Completed")

  rbm.save(new DataOutputStream(new FileOutputStream("/home/alno/nn-rbm.dat")))

  println("Params saved")

}