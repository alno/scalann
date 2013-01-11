package org.scalann

import org.scalatest.FunSpec
import org.scalatest.matchers.ShouldMatchers
import breeze.linalg._
import scala.math._

abstract class BaseSpec extends FunSpec with ShouldMatchers {

  def vec(elems: Double*) = DenseVector(elems: _*)

  val gradientThreshold = 1e-5
  val gradientStep = 1e-3

  val distances = List(-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0)
  val weights = List(1 / 280.0, -4 / 105.0, 0.2, -0.8, 0.8, -0.2, 4 / 105.0, -1 / 280.0)

  implicit def convertToVectorShouldWrapper[T](o: Vector[T]): AnyShouldWrapper[Vector[T]] = new AnyShouldWrapper(o)

  def checkParamGradient(layer: Stage, input: DenseVector[Double], target: DenseVector[Double], gradient: DenseVector[Double]) {
    val oldParams = layer.params.copy

    for (i <- 0 until gradient.size) {
      val offset = DenseVector.zeros[Double](oldParams.size)
      offset(i) = gradientStep

      var temp = 0.0

      for (j <- 0 until distances.size) {
        layer.update(offset * distances(j))
        temp += layer.exampleLoss(input -> target) * weights(j)
        layer.update(oldParams - layer.params)
      }

      gradient(i) should be((temp / gradientStep) plusOrMinus gradientThreshold)
    }
  }

  def chechInputGradient(layer: Stage, input: DenseVector[Double], target: DenseVector[Double], gradient: DenseVector[Double]) {
    for (i <- 0 until gradient.size) {
      val offset = DenseVector.zeros[Double](input.size)
      offset(i) = gradientStep

      var temp = 0.0

      for (j <- 0 until distances.size) {
        temp += layer.exampleLoss((input + offset * distances(j)) -> target) * weights(j)
      }

      gradient(i) should be((temp / gradientStep) plusOrMinus gradientThreshold)
    }
  }

}