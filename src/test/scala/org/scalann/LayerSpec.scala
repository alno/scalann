package org.scalann

import org.scalatest.FunSpec
import org.scalatest.matchers.ShouldMatchers
import breeze.plot._
import breeze.linalg._

class LayerSpec extends FunSpec with ShouldMatchers {
  
  def zv(size: Int) = DenseVector.zeros[Double](size)

  val layer = new BasicLayer(3, 2)
  
  describe("Layer activation") {
    
    val input = zv(3)
    
    it("should produce correctly-sized output") {
      val output = layer(input)
      
      output.size should be(2)
    }
    
    it("should produce correctly-sized derivations") {
      val (output, memo) = layer.forward(zv(3))
      val (dInput, WeightBiasGradient(dWeights, dBiases)) = memo.backward(zv(2))
      
      dInput.size should be(3)
      dWeights.cols should be(3)
      dWeights.rows should be(2)
      dBiases.size should be(2)
    }
    
  }
  
}