package org.scalann.examples

import breeze.linalg._
import nak.classify.Classifier
import org.scalann.Stage

class BreezeClassifier(stage: Stage) extends Classifier[Int, DenseVector[Double]] {

  def scores(vec: DenseVector[Double]) = Counter(stage(vec).activeIterator)

}
