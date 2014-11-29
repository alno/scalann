package org.scalann

import breeze.linalg._
import breeze.numerics._
import org.scalann.loss.LogisticLoss

class LogisticLayer(inputSize: Int, outputSize: Int) extends AbstractLayer(inputSize, outputSize) {

  protected def outputTransform(v: DenseVector[Double]) =
    sigmoid.inPlace(v)

  protected def outputDerivationTransform(dv: DenseVector[Double], v: DenseVector[Double]) {
    val ddata = dv.data
    val dstride = dv.stride

    val vdata = v.data
    val vstride = v.stride

    var dpos = dv.offset
    var vpos = v.offset
    var ind = 0

    while (ind < dv.size) {
      ddata(dpos) *= vdata(vpos) * (1 - vdata(vpos))
      dpos += dstride
      vpos += vstride
      ind += 1
    }
  }

  override def loss = LogisticLoss

}
