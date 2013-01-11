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

  def forward(input: DenseVector[Double]): (DenseVector[Double], Memo) = {
    val memos = new Array[Stage#Memo](layers.size)
    var current = input

    for (i <- 0 until layers.size) {
      val (output, memo) = layers(i).forward(current)

      current = output
      memos(i) = memo
    }

    current -> new Memo {

      override def backwardAdd(derivation: DenseVector[Double], outputDeriv: Boolean)(inputGradAcc: DenseVector[Double], paramGradAcc: DenseVector[Double]) {
        var curOutputDeriv = outputDeriv
        var curDerivation = derivation
        var curEndPos = paramSize
        var i = layers.size - 1

        while (i >= 0) {
          val nextEndPos = curEndPos - layers(i).paramSize
          val nextDerivation = DenseVector.zeros[Double](layers(i).inputSize)

          memos(i).backwardAdd(curDerivation, curOutputDeriv)(nextDerivation, paramGradAcc(nextEndPos until curEndPos))

          curEndPos = nextEndPos
          curDerivation = nextDerivation
          curOutputDeriv = true
          i -= 1
        }

        inputGradAcc += curDerivation
      }

    }
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