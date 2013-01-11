package org.scalann

import breeze.linalg._
import scala.annotation.tailrec

class FeedForwardNetwork(val layers: List[Stage]) extends Stage {

  validateLayers(layers.head, layers.tail)

  val inputSize = layers.head.inputSize
  val outputSize = layers.last.outputSize
  val paramSize = layers.view.map(_.paramSize).sum

  override def apply(input: DenseVector[Double]): DenseVector[Double] =
    layers.foldLeft(input) { (in, layer) => layer(in) }

  def forward(input: DenseVector[Double]): (DenseVector[Double], Memo) =
    forwardThrough(layers, input, Nil)

  @tailrec
  private def forwardThrough(layers: List[Stage], input: DenseVector[Double], memos: List[Stage#Memo]): (DenseVector[Double], Memo) = layers match {
    case Nil =>
      input -> new Memo {

        def backward(derivation: DenseVector[Double], outputDeriv: Boolean = false) =
          backwardThrough(memos, derivation, outputDeriv, Nil)

      }
    case layer :: others =>
      val (output, memo) = layer.forward(input)

      forwardThrough(others, output, memo :: memos)
  }

  @tailrec
  private def backwardThrough(memos: List[Stage#Memo], derivation: DenseVector[Double], outputDeriv: Boolean, gradients: List[DenseVector[Double]]): (DenseVector[Double], DenseVector[Double]) = memos match {
    case Nil =>
      derivation -> DenseVector.vertcat(gradients: _*)

    case memo :: prevMemos =>
      val (prevDerivation, gradient) = memo.backward(derivation, outputDeriv)

      backwardThrough(prevMemos, prevDerivation, true, gradient :: gradients)
  }

  def params =
    DenseVector.vertcat(layers.view.map(_.params): _*)

  def update(gradient: DenseVector[Double]) =
    updateLayers(layers, gradient, 0)

  def cost(actual: DenseVector[Double], target: DenseVector[Double]) =
    layers.last.cost(actual, target)

  @tailrec
  private def updateLayers(layers: List[Stage], gradient: DenseVector[Double], pos: Int): Unit = layers match {
    case Nil =>
      require(pos == gradient.size, "Gradient size should be equal to sum of layer params sizes")
    case layer :: others =>
      layer.update(gradient(pos until (pos + layer.paramSize)))
      updateLayers(layers.tail, gradient, pos + layer.paramSize)
  }

  @tailrec
  private def validateLayers(head: Stage, tail: Traversable[Stage]): Unit =
    if (tail.size > 0) {
      require(head.outputSize == tail.head.inputSize)
      validateLayers(tail.head, tail.tail)
    }

}