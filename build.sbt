import scalann.sbtplugins.Utils._

name := "scalann"

organization := "org.scalann"

version := "0.1.0-SNAPSHOT"

scalaVersion := "2.10.0"

resolvers ++= Seq(
  "Sonatype repository" at "https://oss.sonatype.org/content/repositories/snapshots/"
)

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze-math" % "0.2-SNAPSHOT",
  "org.scalanlp" %% "breeze-viz" % "0.2-SNAPSHOT",
  "org.scalanlp" %% "breeze-learn" % "0.2-SNAPSHOT"
)

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "1.9.1" % "test",
  "com.google.caliper" % "caliper" % "0.5-rc1" % "test",
  "junit" % "junit" % "4.8.2" % "test"
)

caliperRunTask(TaskKey[Unit]("benchmark"), Test, "org.scalann.LearningBench")
