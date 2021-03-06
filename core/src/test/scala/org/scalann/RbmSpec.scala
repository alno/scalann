package org.scalann

import breeze.plot._
import breeze.linalg._
import scala.math._

class RbmSpec extends BaseSpec {

  describe("Rbm with random params") {
    val rbm = new Rbm(3, 2)
    val input = vecRand(3)

    describe("gradient") {
      val gradient = rbm.gradient(input)

      it("should have correct size") {
        gradient.size should be(11)
      }

    }

    describe("should be saved and correctly restored") {
      checkSaveRestore(rbm)
    }

  }

}
