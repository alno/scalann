package org.scalann.examples

import java.io.{ FileInputStream, DataInputStream }
import breeze.linalg._

class MnistLabelReader(fileName: String) {

  private[this] val stream = new DataInputStream(new FileInputStream(fileName))

  assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")

  val count = stream.readInt()

  val labelsAsInts = readLabels(0)
  val labelsAsVectors = labelsAsInts.map { label =>
    DenseVector.tabulate[Double](10) { i => if (i == label) 1.0 else 0.0 }
  }

  private[this] def readLabels(ind: Int): Stream[Int] =
    if (ind >= count)
      Stream.empty
    else
      Stream.cons(stream.readByte(), readLabels(ind + 1))

}

class MnistImageReader(fileName: String) {

  private[this] val stream = new DataInputStream(new FileInputStream(fileName))

  assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")

  val count = stream.readInt()
  val width = stream.readInt()
  val height = stream.readInt()

  val imagesAsMatrices = readImages(0)
  val imagesAsVectors = imagesAsMatrices map { image =>
    DenseVector.tabulate(width * height) { i => image(i / width, i % height) / 255.0 }
  }

  private[this] def readImages(ind: Int): Stream[DenseMatrix[Int]] =
    if (ind >= count)
      Stream.empty
    else
      Stream.cons(readImage(), readImages(ind + 1))

  private[this] def readImage(): DenseMatrix[Int] = {
    val m = DenseMatrix.zeros[Int](height, width)

    for (y <- 0 until height; x <- 0 until width)
      m(y, x) = stream.readUnsignedByte()

    m
  }

}

class Mnist(location: String) {

  lazy val imageReader = new MnistImageReader(location + "/train-images-idx3-ubyte")
  lazy val labelReader = new MnistLabelReader(location + "/train-labels-idx1-ubyte")

  def imageWidth = imageReader.width
  def imageHeight = imageReader.height

  def imagesAsMatrices = imageReader.imagesAsMatrices
  def imagesAsVectors = imageReader.imagesAsVectors

  def labelsAsInts = labelReader.labelsAsInts
  def labelsAsVectors = labelReader.labelsAsVectors

  def examples = imagesAsVectors zip labelsAsVectors

}