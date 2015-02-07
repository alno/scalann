package org.scalann.builder

import org.scalann.stages.LinearLayer

case class Linear(size: Int) extends AbstractLayerBlueprint[LinearLayer](size) {

  override def buildForInputs(inputSize: Int) =
    new LinearLayer(inputSize, size)

}
