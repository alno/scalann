package org.scalann

import breeze.linalg._
import breeze.classify.Classifier

class BreezeClassifier(stage: Stage) extends Classifier[Int, DenseVector[Double]] {

  def scores(vec: DenseVector[Double]) = Counter(stage(vec).activeIterator)

}
