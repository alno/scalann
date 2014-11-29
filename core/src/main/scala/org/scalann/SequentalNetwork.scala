package org.scalann

import breeze.linalg._
import scala.annotation.tailrec

class SequentalNetwork(val layers: List[Stage]) extends Stage {

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

      override def backwardAdd(derivation: DenseVector[Double], outputDeriv: Boolean)(inputGradAcc: DenseVector[Double], inputFactor: Double, paramGradAcc: DenseVector[Double], paramFactor: Double) {
        var curOutputDeriv = outputDeriv
        var curDerivation = derivation
        var curEndPos = paramSize
        var i = layers.size - 1

        while (i > 0) {
          val nextEndPos = curEndPos - layers(i).paramSize
          val nextDerivation = DenseVector.zeros[Double](layers(i).inputSize)

          memos(i).backwardAdd(curDerivation, curOutputDeriv)(nextDerivation, 1.0, paramGradAcc(nextEndPos until curEndPos), paramFactor)

          curEndPos = nextEndPos
          curDerivation = nextDerivation
          curOutputDeriv = true
          i -= 1
        }

        if (layers.size > 0) {
          memos(0).backwardAdd(curDerivation, curOutputDeriv)(inputGradAcc, inputFactor, paramGradAcc(0 until curEndPos), paramFactor)
        }
      }

    }
  }

  def params =
    DenseVector.vertcat(layers.view.map(_.params): _*)

  def paramsDecay =
    DenseVector.vertcat(layers.view.map(_.paramsDecay): _*)

  def updateParams(gradient: DenseVector[Double]) =
    layers.foldLeft(0) { (pos, layer) =>
      layer.updateParams(gradient(pos until (pos + layer.paramSize)))
      pos + layer.paramSize
    }

  def assignParams(newParams: DenseVector[Double]) =
    layers.foldLeft(0) { (pos, layer) =>
      layer.assignParams(newParams(pos until (pos + layer.paramSize)))
      pos + layer.paramSize
    }

  override def loss = layers.last.loss

  @tailrec
  private def validateLayers(head: Stage, tail: Traversable[Stage]): Unit =
    if (tail.size > 0) {
      require(head.outputSize == tail.head.inputSize)
      validateLayers(tail.head, tail.tail)
    }

}