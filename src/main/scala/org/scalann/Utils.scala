package org.scalann

import breeze.generic.UFunc
import breeze.linalg._
import scala.math._

object Utils {

  def sample = UFunc { (x: Double) =>
    if (x > random) 1.0 else 0.0
  }

}
