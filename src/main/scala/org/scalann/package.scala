package org

package object scalann {

  implicit def stageIo(stage: Stage) = new StageIO(stage)

}