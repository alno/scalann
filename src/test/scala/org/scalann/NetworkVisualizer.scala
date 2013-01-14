package org.scalann

import breeze.linalg._
import breeze.plot._
import java.io.DataInputStream
import java.io.FileInputStream
import java.awt.Graphics2D

object NetworkVisualizer extends App {

  val w, h = 28
  val ll = new LogisticLayer(w * h, 50)
  val nn = new FeedForwardNetwork(List(ll, new SoftmaxLayer(50, 10)))

  nn.restore(new DataInputStream(new FileInputStream("/home/alno/nn-simple-wd.dat")))

  ExportGraphics.writeFile(
    new java.io.File("/home/alno/image-wd.png"),
    draw = drawPlots(ll.weights),
    width = 1000,
    height = 500)

  println("nn saved")

  val rbm = new Rbm(w * h, 50)

  rbm.restore(new DataInputStream(new FileInputStream("/home/alno/nn-rbm.dat")))

  ExportGraphics.writeFile(
    new java.io.File("/home/alno/image-rbm.png"),
    draw = drawPlots(rbm.weights),
    width = 1000,
    height = 500)

  println("rbm saved")

  def drawPlots(weights: DenseMatrix[Double])(g2d: Graphics2D) {
    val plotWidth = 100
    val plotHeight = 100

    for (x <- 0 until 10)
      for (y <- 0 until 5) {
        val i = x * 5 + y
        val row = weights(i, ::).copy
        val img = image(new DenseMatrix(w, h, row.data, row.offset))
        val plot = new Plot

        plot += img
        plot.chart.draw(g2d, new java.awt.Rectangle(x * plotWidth, y * plotHeight, plotWidth, plotHeight))
      }
  }
}