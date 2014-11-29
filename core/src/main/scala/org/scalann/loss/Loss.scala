package org.scalann.loss

import breeze.linalg.Vector

trait Loss extends ((Vector[Double], Vector[Double]) => Double)
