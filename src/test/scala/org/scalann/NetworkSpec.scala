package org.scalann

import org.scalatest.FunSpec
import org.scalatest.matchers.ShouldMatchers
import breeze.plot._
import breeze.linalg._
import scala.math._

class NetworkSpec extends BaseSpec {

  def zv(size: Int) = DenseVector.fill[Double](size)(math.random)

  val testLayerConfigs = Map(
    new FeedForwardNetwork(List(new LogisticLayer(4, 3), new LogisticLayer(3, 2))) -> logisticTarget(2),
    new FeedForwardNetwork(List(new LogisticLayer(4, 3), new SoftmaxLayer(3, 2))) -> softmaxTarget(2),
    new FeedForwardNetwork(List(new LogisticLayer(4, 3), new LinearLayer(3, 2))) -> linearTarget(2))

  testLayerConfigs.foreach {
    case (net, target) =>
      describe(net.getClass.getSimpleName + "(" + net.layers.map(_.getClass.getSimpleName).mkString(", ") + ") with random params") {
        val input = zv(4)

        describe("output") {
          val output = net(input)

          it("should have correct size") {
            output.size should be(2)
          }
        }

        describe("derivations") {
          val (output, memo) = net.forward(input)
          val (dInput, dParam) = memo.backward(output - target)

          it("should have correct size") {
            dInput.size should be(4)
            dParam.size should be(23)
          }

          it("should be correct (in params)") {
            checkParamGradient(net, input, target, dParam)
          }

          it("should be correct (in input)") {
            chechInputGradient(net, input, target, dInput)
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
