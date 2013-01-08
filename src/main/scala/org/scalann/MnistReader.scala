package org.scalann

import java.io.FileInputStream
import java.io.DataInputStream
import breeze.linalg.DenseMatrix

class MnistLabelsReader(fileName: String) {

  private[this] val stream = new DataInputStream(new FileInputStream(fileName))

  assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")

  val count = stream.readInt()

  val labels = readLabels(0)

  private[this] def readLabels(ind: Int): Stream[Int] =
    if (ind >= count)
      Stream.empty
    else
      Stream.cons(stream.readByte(), readLabels(ind + 1))

}

class MnistImagesReader(fileName: String) {

  private[this] val stream = new DataInputStream(new FileInputStream(fileName))

  assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")

  val count = stream.readInt()
  val width = stream.readInt()
  val height = stream.readInt()

  val images = readImages(0)

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

object MnistReader {

  val imagesReader = new MnistImagesReader("/home/alno/mnist/train-images-idx3-ubyte")
  val labelsReader = new MnistLabelsReader("/home/alno/mnist/train-labels-idx1-ubyte")

}