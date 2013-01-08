package org.scalann

import org.scalatest.FunSpec
import org.scalatest.matchers.ShouldMatchers
import breeze.plot._
import breeze.linalg._

class MnistReaderSpec extends FunSpec with ShouldMatchers {

  describe("MnistImagesReader") {

    val reader = MnistReader.imagesReader

    it("should read correct image count") {
      reader.count should be(60000)
    }

    it("should read correct image width") {
      reader.width should be(28)
    }

    it("should read correct image height") {
      reader.height should be(28)
    }

    it("should read correct images") {
      reader.images.head.map(_ / 255.0)
    }
  }

  describe("MnistLabelsReader") {
    val reader = MnistReader.labelsReader

    it("should read correct label count") {
      reader.count should be(60000)
    }

    it("should read correct labels") {
      reader.labels.take(10) should be(Stream(5, 0, 4, 1, 9, 2, 1, 3, 1, 4))
    }

  }

}
