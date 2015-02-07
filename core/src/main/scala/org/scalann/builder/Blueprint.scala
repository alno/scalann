package org.scalann.builder

import org.scalann.Stage
import org.scalann.stages.SeqStage
import org.scalann.stages.AbstractLayer

trait Blueprint[T <: Stage] {

  def :>:[H <: Stage](head: Blueprint[H]) = Blueprint.:>:(head, this)
  def :>:(input: Input) = buildForInputs(input.size)

  def buildForInputs(inputSize: Int): T

  def calculateParamSize(inputSize: Int): Int

}

abstract class AbstractLayerBlueprint[T <: AbstractLayer](outputSize: Int) extends Blueprint[T] {

  override def calculateParamSize(inputSize: Int): Int = outputSize * (inputSize + 1)

}

object Blueprint {

  case class :>:[H <: Stage, T <: Stage](head: Blueprint[H], tail: Blueprint[T]) extends Blueprint[SeqStage[H, T]] {

    override def buildForInputs(inputSize: Int) = {
      val headStage = head.buildForInputs(inputSize)
      val tailStage = tail.buildForInputs(headStage.outputSize)
      new SeqStage(headStage, tailStage)
    }

    override def calculateParamSize(inputSize: Int): Int = {
      val headParamSize = head.calculateParamSize(inputSize)
      val tailParamSize = tail.calculateParamSize(inputSize)

      headParamSize + tailParamSize
    }

  }

}
