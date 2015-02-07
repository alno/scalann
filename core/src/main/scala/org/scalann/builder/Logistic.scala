package org.scalann.builder

import org.scalann.stages.LogisticLayer

case class Logistic(size: Int) extends AbstractLayerBlueprint[LogisticLayer](size) {

  override def buildForInputs(inputSize: Int) =
    new LogisticLayer(inputSize, size)

}
