package org.scalann

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
    new LogisticLayer(3, 2) -> logisticTarget _,
    new SoftmaxLayer(3, 2) -> softmaxTarget _,
    new LinearLayer(3, 2) -> linearTarget _)

  testLayerConfigs.foreach {
    case (layer, targetFun) =>
      describe(layer.getClass.getSimpleName + " with random params") {

        describe("output") {
          val input = zv(3)
          val output = layer(input)

          it("should have correct size") {
            output.size should be(2)
          }
        }

        describe("derivations returned by backward") {
          val input = zv(3)
          val target = targetFun(2)
          val (output, memo) = layer.forward(input)
          val (dInput, dParam) = memo.backward(output - target)

          it("should have correct size") {
            dInput.size should be(3)
            dParam.size should be(8)
          }

          it("should be correct (in params)") {
            checkParamGradient(layer, input, target, dParam, 1.0)
          }

          it("should be correct (in input)") {
            checkInputGradient(layer, input, target, dInput, 1.0)
          }

        }

        describe("derivations returned by backwardAdd") {
          val input = zv(3)
          val target = targetFun(2)
          val data = Array.fill(100)(0.0)
          val dInput = new DenseVector(data, 10, 1, 3)
          val dParam = new DenseVector(data, 20, 1, 8)

          val (output, memo) = layer.forward(input)

          memo.backwardAdd(output - target, false)(dInput, 0.4, dParam, 0.7)

          it("should have correct size") {
            dInput.size should be(3)
            dParam.size should be(8)
          }

          it("should be correct (in input)") {
            checkInputGradient(layer, input, target, dInput, 0.4)
          }

          it("should be correct (in params)") {
            checkParamGradient(layer, input, target, dParam, 0.7)
          }

        }

        describe("derivations returned by gradient with multiple examples") {
          val inputs = List.fill(10) { zv(3) }
          val targets = List.fill(10) { targetFun(2) }

          val gradient = layer.gradient(inputs zip targets)

          it("should be correct (in params)") {
            checkParamGradientMulti(layer, inputs, targets, gradient)
          }

        }

        describe("should be saved and correctly restored") {
          checkSaveRestore(layer)
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
