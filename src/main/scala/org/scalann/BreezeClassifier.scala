package org.scalann

import breeze.classify.Classifier
import breeze.linalg.DenseVector
import breeze.linalg.Counter

class BreezeClassifier(stage: Stage) extends Classifier[Int, DenseVector[Double]] {

  def scores(vec: DenseVector[Double]) = Counter(stage(vec).activeIterator)

}
