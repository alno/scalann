import sbt._
import Keys._

object common {
  val settings = Seq(
    scalaVersion := "2.10.4",
    version := "0.2.0-SNAPSHOT",
    organization := "org.scalann"
  )

  val breezeVersion = "0.2.3"

  val scalaTestVersion = "2.2.1"
}
