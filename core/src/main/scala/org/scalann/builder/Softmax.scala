package org.scalann.builder

import org.scalann.stages.SoftmaxLayer

case class Softmax(size: Int) extends AbstractLayerBlueprint[SoftmaxLayer](size) {

  override def buildForInputs(inputSize: Int) =
    new SoftmaxLayer(inputSize, size)

}
