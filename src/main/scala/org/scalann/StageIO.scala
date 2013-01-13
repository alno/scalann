package org.scalann

import java.io.DataInput
import java.io.DataOutput
import breeze.linalg._

class StageIO(stage: Stage) {

  def restore(in: DataInput): Unit =
    stage.assignParams(DenseVector.fill(stage.paramSize) { in.readDouble })

  def save(out: DataOutput): Unit =
    stage.params.foreach(out.writeDouble)

}