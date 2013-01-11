package org.scalann

import org.scalatest.FunSpec
import org.scalatest.matchers.ShouldMatchers
import breeze.plot._
import breeze.linalg._
import scala.math._

class LayerSpec extends BaseSpec {

  def zv(size: Int) = DenseVector.fill[Double](size)(math.random)
  def sqr(x: Double) = x * x
  val e1 = exp(-1)

  describe("LogisticLayer with zero weights and ident biases") {
    val layer = new LogisticLayer(3, 2)
    layer.params := vec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)

    it("should have mean output") {
      layer(zv(3)) should be(vec(1 / (1 + e1), 1 / (1 + e1)))
    }

    it("should have zero input gradient") {
      layer.forward(zv(3))._2.backward(zv(2))._1 should be(vec(0.0, 0.0, 0.0))
    }

    it("should have correct bias gradient") {
      layer.forward(zv(3))._2.backward(vec(0.5, -0.5))._2(6 to 7) should be(vec(0.5, -0.5))
    }

  }

  describe("SoftmaxLayer with zero weights and ident biases") {
    val layer = new SoftmaxLayer(3, 2)
    layer.params := vec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)

    it("should have correct output") {
      layer(zv(3)) should be(vec(0.5, 0.5))
    }

    it("should have zero input gradient") {
      layer.forward(zv(3))._2.backward(zv(2))._1 should be(vec(0.0, 0.0, 0.0))
    }

    it("should have correct bias gradient") {
      layer.forward(zv(3))._2.backward(vec(0.5, -0.5))._2(6 to 7) should be(vec(0.5, -0.5))
    }

  }

  List(new LogisticLayer(3, 2), new SoftmaxLayer(3, 2)).foreach { layer =>
    describe(layer.getClass.getSimpleName + " with zero params") {
      layer.params := vec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

      it("should have mean output") {
        layer(zv(3)) should be(vec(0.5, 0.5))
      }

      it("should have zero input gradient") {
        layer.forward(zv(3))._2.backward(zv(2))._1 should be(vec(0.0, 0.0, 0.0))
      }

    }
  }

  val testLayerConfigs = Map(
    new LogisticLayer(3, 2) -> logisticTarget(2),
    new SoftmaxLayer(3, 2) -> softmaxTarget(2),
    new LinearLayer(3, 2) -> linearTarget(2))

  testLayerConfigs.foreach {
    case (layer, target) =>
      describe(layer.getClass.getSimpleName + " with random params") {
        val input = zv(3)

        describe("output") {
          val output = layer(input)

          it("should have correct size") {
            output.size should be(2)
          }
        }

        describe("derivations") {
          val (output, memo) = layer.forward(input)
          val (dInput, dParam) = memo.backward(output - target)

          it("should have correct size") {
            dInput.size should be(3)
            dParam.size should be(8)
          }

          it("should be correct (in params)") {
            checkParamGradient(layer, input, target, dParam)
          }

          it("should be correct (in input)") {
            chechInputGradient(layer, input, target, dInput)
          }

        }

      }

  }

  def softmaxTarget(size: Int) = {
    val target = DenseVector.zeros[Double](size)
    target((math.random * target.size).toInt) = 1.0
    target
  }

  def logisticTarget(size: Int) =
    DenseVector.fill[Double](size)(math.random)

  def linearTarget(size: Int) =
    DenseVector.fill[Double](size)(math.random * 4 - 2)

}
