import sbt._
import Keys._

object common {
  val settings = Seq(
    scalaVersion := "2.11.4",
    version := "0.2.0-SNAPSHOT",
    organization := "org.scalann"
  )

  val breezeVersion = "0.9"
  val breezeVizVersion = "0.9"
  val nakVersion = "1.3"

  val scalaTestVersion = "2.2.1"
}
