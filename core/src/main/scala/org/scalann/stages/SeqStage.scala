package org.scalann.stages

import breeze.linalg._
import org.scalann.Stage

case class SeqStage[+H <: Stage, +T <: Stage](head: H, tail: T) extends Stage with Serializable {

  require(head.outputSize == tail.inputSize)

  val inputSize = head.inputSize
  val outputSize = tail.outputSize

  val paramSize = head.paramSize + tail.paramSize

  override def toString = s"$head :>: $tail"

  override def apply(input: DenseVector[Double]): DenseVector[Double] =
    tail(head(input))

  def forward(input: DenseVector[Double]): (DenseVector[Double], Memo) = {
    val (headOutput, headMemo) = head.forward(input)
    val (tailOutput, tailMemo) = tail.forward(headOutput)

    tailOutput -> new Memo {

      override def backwardAdd(derivation: DenseVector[Double], outputDeriv: Boolean)(inputGradAcc: DenseVector[Double], inputFactor: Double, paramGradAcc: DenseVector[Double], paramFactor: Double) {
        val nextDerivation = DenseVector.zeros[Double](tail.inputSize)

        tailMemo.backwardAdd(derivation, outputDeriv)(nextDerivation, 1.0, paramGradAcc(head.paramSize until paramSize), paramFactor)
        headMemo.backwardAdd(nextDerivation, true)(inputGradAcc, inputFactor, paramGradAcc(0 until head.paramSize), paramFactor)
      }

    }
  }

  def params =
    DenseVector.vertcat(head.params, tail.params)

  def paramsDecay =
    DenseVector.vertcat(head.paramsDecay, tail.paramsDecay)

  // TODO Remove
  def updateParams(gradient: DenseVector[Double]) = {
    require(gradient.size == paramSize)

    head.updateParams(gradient(0 until head.paramSize))
    tail.updateParams(gradient(head.paramSize until gradient.size))
  }

  // TODO Remove
  def assignParams(newParams: DenseVector[Double]) = {
    require(newParams.size == paramSize)

    head.assignParams(newParams(0 until head.paramSize))
    tail.assignParams(newParams(head.paramSize until newParams.size))
  }

  override def loss = tail.loss

}