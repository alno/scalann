package org.scalann

import breeze.generic.UFunc
import breeze.linalg._
import scala.math._

object Utils {

  def sample = UFunc { (x: Double) =>
    if (x > fastRandomDouble) 1.0 else 0.0
  }

  def zero = UFunc { (x: Any) => 0.0 }

  private var randomState: Long = System.nanoTime

  def fastRandomDouble = (fastRandomLong & 0xFFFF) / 65536.0

  def fastRandomLong: Long = {
    randomState ^= (randomState << 21)
    randomState ^= (randomState >>> 35)
    randomState ^= (randomState << 4)
    randomState
  }

}
