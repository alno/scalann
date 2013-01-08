name := "scalann"

organization := "org.scalann"

version := "0.1.0-SNAPSHOT"

scalaVersion := "2.10.0"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze-math" % "0.2-SNAPSHOT",
  "org.scalanlp" %% "breeze-viz" % "0.2-SNAPSHOT"
)

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "1.8" % "test",
  "junit" % "junit" % "4.8.2" % "test"
)
