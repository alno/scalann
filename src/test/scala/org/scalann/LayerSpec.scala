package org.scalann

import org.scalatest.FunSpec
import org.scalatest.matchers.ShouldMatchers
import breeze.plot._
import breeze.linalg._
import scala.math._

class LayerSpec extends FunSpec with ShouldMatchers {

  def zv(size: Int) = DenseVector.fill[Double](size)(math.random)
  def sqr(x: Double) = x * x
  val e1 = exp(-1)

  describe("LogisticLayer with zero weights and ident biases") {
    val layer = new LogisticLayer(3, 2)
    layer.params := DenseVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)

    it("should have mean output") {
      layer(zv(3)).data should be(Array(1 / (1 + e1), 1 / (1 + e1)))
    }

    it("should have zero input gradient") {
      layer.forward(zv(3))._2.backward(zv(2))._1.data should be(Array(0.0, 0.0, 0.0))
    }

    it("should have correct bias gradient") {
      layer.forward(zv(3))._2.backward(DenseVector(0.5, -0.5))._2(6 to 7).copy.data should be(Array(0.5, -0.5))
    }

  }

  describe("SoftmaxLayer with zero weights and ident biases") {
    val layer = new SoftmaxLayer(3, 2)
    layer.params := DenseVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)

    it("should have correct output") {
      layer(zv(3)).data should be(Array(0.5, 0.5))
    }

    it("should have zero input gradient") {
      layer.forward(zv(3))._2.backward(zv(2))._1.data should be(Array(0.0, 0.0, 0.0))
    }

    it("should have correct bias gradient") {
      layer.forward(zv(3))._2.backward(DenseVector(0.5, -0.5))._2(6 to 7).copy.data should be(Array(0.5, -0.5))
    }

  }

  List(new LogisticLayer(3, 2), new SoftmaxLayer(3, 2)).foreach { layer =>
    describe(layer.getClass.getSimpleName + " with zero params") {
      layer.params := DenseVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

      it("should have mean output") {
        layer(zv(3)).data should be(Array(0.5, 0.5))
      }

      it("should have zero input gradient") {
        layer.forward(zv(3))._2.backward(zv(2))._1.data should be(Array(0.0, 0.0, 0.0))
      }

    }
  }

  List(new LogisticLayer(3, 2), new SoftmaxLayer(3, 2)).foreach { layer =>
    describe(layer.getClass.getSimpleName + " with random params") {
      val input = zv(3)
      val target = DenseVector.zeros[Double](2)
      target((math.random * target.size).toInt) = 1.0

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

  val threshold = 1e-5
  val step = 1e-3

  val distances = List(-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0)
  val weights = List(1 / 280.0, -4 / 105.0, 0.2, -0.8, 0.8, -0.2, 4 / 105.0, -1 / 280.0)

  def checkParamGradient(layer: Stage, input: DenseVector[Double], target: DenseVector[Double], gradient: DenseVector[Double]) {
    val oldParams = layer.params.copy

    for (i <- 0 until gradient.size) {
      val offset = DenseVector.zeros[Double](oldParams.size)
      offset(i) = step

      var temp = 0.0

      for (j <- 0 until distances.size) {
        layer.update(offset * distances(j))
        temp += layer.exampleLoss(input -> target) * weights(j)
        layer.update(oldParams - layer.params)
      }

      gradient(i) should be((temp / step) plusOrMinus threshold)
    }
  }

  def chechInputGradient(layer: Stage, input: DenseVector[Double], target: DenseVector[Double], gradient: DenseVector[Double]) {
    for (i <- 0 until gradient.size) {
      val offset = DenseVector.zeros[Double](input.size)
      offset(i) = step

      var temp = 0.0

      for (j <- 0 until distances.size) {
        temp += layer.exampleLoss((input + offset * distances(j)) -> target) * weights(j)
      }

      gradient(i) should be((temp / step) plusOrMinus threshold)
    }
  }

}
