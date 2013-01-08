package org.scalann

import breeze.linalg._
import scala.annotation.tailrec

case class FeedForwardNetworkGradient(layerGradients: List[Gradient]) extends Gradient {

  def +=(that: Gradient) = that match {
    case FeedForwardNetworkGradient(thatLayerGradients) =>
      require(layerGradients.size == thatLayerGradients.size)

      mergeGradients(layerGradients, thatLayerGradients)
  }

  def *=(factor: Double) {
    layerGradients.foreach(_ *= factor)
  }

  @tailrec
  private def mergeGradients(gradients: List[Gradient], thatGradients: List[Gradient]): Unit =
    if (gradients.isEmpty) {
      require(thatGradients.isEmpty)
    } else {
      gradients.head += thatGradients.head
      mergeGradients(gradients.tail, thatGradients.tail)
    }

}

class FeedForwardNetwork(layers: List[Stage]) extends Stage {

  validateLayers(layers.head, layers.tail)

  def inputSize = layers.head.inputSize
  def outputSize = layers.last.outputSize

  override def apply(input: DenseVector[Double]): DenseVector[Double] =
    layers.foldLeft(input) { (in, layer) => layer(in) }

  def forward(input: DenseVector[Double]): (DenseVector[Double], Memo) =
    forwardThrough(layers, input, Nil)

  @tailrec
  private def forwardThrough(layers: List[Stage], input: DenseVector[Double], memos: List[Stage#Memo]): (DenseVector[Double], Memo) = layers match {
    case Nil =>
      input -> new Memo {

        def backward(derivation: DenseVector[Double]) =
          backwardThrough(memos, derivation, Nil)

      }
    case layer :: others =>
      val (output, memo) = layer.forward(input)

      forwardThrough(others, output, memo :: memos)
  }

  @tailrec
  private def backwardThrough(memos: List[Stage#Memo], derivation: DenseVector[Double], gradients: List[Gradient]): (DenseVector[Double], Gradient) = memos match {
    case Nil =>
      derivation -> FeedForwardNetworkGradient(gradients)

    case memo :: prevMemos =>
      val (prevDerivation, gradient) = memo.backward(derivation)

      backwardThrough(prevMemos, prevDerivation, gradient :: gradients)
  }

  def update(grad: Gradient): Unit = grad match {
    case FeedForwardNetworkGradient(layerGradients) =>
      require(layerGradients.size == layers.size)

      updateLayers(layers, layerGradients)
  }

  @tailrec
  private def updateLayers(layers: List[Stage], gradients: List[Gradient]): Unit =
    if (layers.isEmpty) {
      require(gradients.isEmpty)
    } else {
      layers.head.update(gradients.head)
      updateLayers(layers.tail, gradients.tail)
    }

  @tailrec
  private def validateLayers(head: Stage, tail: Traversable[Stage]): Unit =
    if (tail.size > 0) {
      require(head.outputSize == tail.head.inputSize)
      validateLayers(tail.head, tail.tail)
    }

}