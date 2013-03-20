package org.scalann

import org.scalatest.FunSpec
import org.scalatest.matchers.ShouldMatchers
import breeze.plot._
import breeze.linalg._
import scala.math._

class NetworkSpec extends BaseSpec {

  def zv(size: Int) = DenseVector.fill[Double](size)(math.random)

  val testLayerConfigs = Map(
    new SequentalNetwork(List(new LogisticLayer(4, 3), new LogisticLayer(3, 2))) -> logisticTarget _,
    new SequentalNetwork(List(new LogisticLayer(4, 3), new SoftmaxLayer(3, 2))) -> softmaxTarget _,
    new SequentalNetwork(List(new LogisticLayer(4, 3), new LinearLayer(3, 2))) -> linearTarget _)

  testLayerConfigs.foreach {
    case (net, targetFun) =>
      describe(net.getClass.getSimpleName + "(" + net.layers.map(_.getClass.getSimpleName).mkString(", ") + ") with random params") {

        describe("output") {
          val target = targetFun(2)
          val input = zv(4)
          val output = net(input)

          it("should have correct size") {
            output.size should be(2)
          }
        }

        describe("derivations") {
          val target = targetFun(2)
          val input = zv(4)
          val (output, memo) = net.forward(input)
          val (dInput, dParam) = memo.backward(output - target)

          it("should have correct size") {
            dInput.size should be(4)
            dParam.size should be(23)
          }

          it("should be correct (in params)") {
            checkParamGradient(net, input, target, dParam, 1.0)
          }

          it("should be correct (in input)") {
            checkInputGradient(net, input, target, dInput, 1.0)
          }

        }

        describe("derivations returned by backwardAdd") {
          val target = targetFun(2)
          val input = zv(4)
          val data = Array.fill(100)(0.0)
          val dInput = new DenseVector(data, 10, 1, 4)
          val dParam = new DenseVector(data, 20, 1, 23)

          val (output, memo) = net.forward(input)

          memo.backwardAdd(output - target, false)(dInput, 0.3, dParam, 0.6)

          it("should have correct size") {
            dInput.size should be(4)
            dParam.size should be(23)
          }

          it("should be correct (in input)") {
            checkInputGradient(net, input, target, dInput, 0.3)
          }

          it("should be correct (in params)") {
            checkParamGradient(net, input, target, dParam, 0.6)
          }

        }

        describe("derivations returned by gradient with multiple examples") {
          val inputs = List.fill(10) { zv(4) }
          val targets = List.fill(10) { targetFun(2) }

          val gradient = net.gradient(inputs zip targets)

          it("should be correct (in params)") {
            checkParamGradientMulti(net, inputs, targets, gradient)
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
