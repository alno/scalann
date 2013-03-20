package org.scalann

import breeze.linalg.DenseVector
import breeze.generic.UFunc
import scala.math._
import java.io.{ DataInput, DataOutput }

package object utils {

  def sample = UFunc { (x: Double) =>
    if (x > fastRandomDouble) 1.0 else 0.0
  }

  /**
   * log(p/(1-p)) = log(p) - log(1-p)
   */
  def logp = UFunc { (p: Double) =>
    log(max(p, 1e-10)) - log(max(1 - p, 1e-10))
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

  implicit class IndexedSeqExt[T](val seq: IndexedSeq[T]) extends AnyVal {

    def sample(count: Int): Seq[T] = {
      val rand = new java.util.Random

      List.fill(count) {
        seq(rand.nextInt(seq.size))
      }
    }

  }

  implicit class ParametrizedIOExt(val stage: Parametrized) extends AnyVal {

    def restore(in: DataInput): Unit =
      stage.assignParams(DenseVector.fill(stage.paramSize) { in.readDouble })

    def save(out: DataOutput): Unit =
      stage.params.foreach(out.writeDouble)

  }

}