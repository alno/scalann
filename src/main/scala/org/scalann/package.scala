package org

package object scalann {

  implicit def stageIo(stage: Parametrized) = new StageIO(stage)

}