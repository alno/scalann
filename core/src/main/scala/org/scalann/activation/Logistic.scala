package org.scalann.activation

import breeze.linalg._
import breeze.numerics._

object Logistic extends ActivationTransform {

  override def transformOutput(v: DenseVector[Double]): Unit =
    sigmoid.inPlace(v)

  override def transformOutputDerivation(dv: DenseVector[Double], v: DenseVector[Double]): Unit = {
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

}
