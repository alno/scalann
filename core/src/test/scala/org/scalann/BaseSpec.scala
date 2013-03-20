package org.scalann

import org.scalann.utils._
import org.scalatest.FunSpec
import org.scalatest.matchers.ShouldMatchers
import breeze.linalg._
import scala.math._

import java.io.{ DataOutputStream, DataInputStream, ByteArrayOutputStream, ByteArrayInputStream }

abstract class BaseSpec extends FunSpec with ShouldMatchers {

  def vec(elems: Double*) = DenseVector(elems: _*)
  def vecRand(size: Int) = DenseVector.fill[Double](size)(math.random)

  val gradientThreshold = 1e-5
  val gradientStep = 1e-3

  val distances = List(-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0)
  val weights = List(1 / 280.0, -4 / 105.0, 0.2, -0.8, 0.8, -0.2, 4 / 105.0, -1 / 280.0)

  implicit def convertToVectorShouldWrapper[T](o: Vector[T]): AnyShouldWrapper[Vector[T]] = new AnyShouldWrapper(o)

  def checkParamGradientMulti(layer: Stage, inputs: Traversable[DenseVector[Double]], targets: Traversable[DenseVector[Double]], gradient: DenseVector[Double]) {
    val oldParams = layer.params.copy

    for (i <- 0 until gradient.size) {
      val offset = DenseVector.zeros[Double](oldParams.size)
      offset(i) = gradientStep

      var temp = 0.0

      for (j <- 0 until distances.size) {
        layer.updateParams(offset * distances(j))
        temp += layer.examplesLoss(inputs.toList zip targets.toList) * weights(j)
        layer.assignParams(oldParams)
      }

      gradient(i) should be((temp / gradientStep) plusOrMinus gradientThreshold)
    }
  }

  def checkParamGradient(layer: Stage, input: DenseVector[Double], target: DenseVector[Double], gradient: DenseVector[Double], factor: Double) {
    val oldParams = layer.params.copy

    for (i <- 0 until gradient.size) {
      val offset = DenseVector.zeros[Double](oldParams.size)
      offset(i) = gradientStep

      var temp = 0.0

      for (j <- 0 until distances.size) {
        layer.updateParams(offset * distances(j))
        temp += layer.exampleLoss(input -> target) * weights(j) * factor
        layer.assignParams(oldParams)
      }

      gradient(i) should be((temp / gradientStep) plusOrMinus gradientThreshold)
    }
  }

  def checkInputGradient(layer: Stage, input: DenseVector[Double], target: DenseVector[Double], gradient: DenseVector[Double], factor: Double) {
    for (i <- 0 until gradient.size) {
      val offset = DenseVector.zeros[Double](input.size)
      offset(i) = gradientStep

      var temp = 0.0

      for (j <- 0 until distances.size) {
        temp += layer.exampleLoss((input + offset * distances(j)) -> target) * weights(j) * factor
      }

      gradient(i) should be((temp / gradientStep) plusOrMinus gradientThreshold)
    }
  }

  def checkSaveRestore(layer: Parametrized) {
    val baos = new ByteArrayOutputStream
    val oldParams = layer.params.copy

    layer.save(new DataOutputStream(baos))
    layer.assignParams(DenseVector.fill(layer.paramSize) { math.random * 10 - 5 })
    layer.params should not be (oldParams)

    layer.restore(new DataInputStream(new ByteArrayInputStream(baos.toByteArray)))
    layer.params should be(oldParams)
  }

}