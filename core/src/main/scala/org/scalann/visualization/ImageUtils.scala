package org.scalann.visualization

import breeze.linalg._
import breeze.plot._
import java.awt.{ Graphics2D, Color, Rectangle }

object ImageUtils {

  def saveMatrices(path: String, inputWidth: Int, inputHeight: Int, layerCols: Int, layerRows: Int)(matrixFun: (Int, Int) => DenseMatrix[Double]) {
    val border = 10
    val padding = 50

    def drawMatrix(weights: DenseMatrix[Double])(g2d: Graphics2D, startX: Int, startY: Int) {
      val img = image(weights)
      val plot = new Plot

      plot += img
      plot.chart.draw(g2d, new Rectangle(startX, startY, inputWidth + padding, inputHeight + padding))
    }

    def drawMatrices(g2d: Graphics2D) =
      for (x <- 0 until layerCols)
        for (y <- 0 until layerRows)
          drawMatrix(matrixFun(x, y))(g2d, border + x * (inputWidth + border + padding), border + y * (inputHeight + border + padding))

    ExportGraphics.writeFile(
      new java.io.File(path),
      draw = drawMatrices,
      width = border + layerCols * (inputWidth + border + padding),
      height = border + layerRows * (inputHeight + border + padding))
  }

  def saveLayerWeight(weights: DenseMatrix[Double], path: String, inputWidth: Int, inputHeight: Int, layerCols: Int, layerRows: Int) =
    saveMatrices(path, inputWidth, inputHeight, layerCols, layerRows) { (x, y) =>
      val arr = weights.t(::, y * layerCols + x).toArray
      new DenseMatrix(inputHeight, inputWidth, arr)
    }

}