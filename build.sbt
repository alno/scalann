name := "scalann"

organization := "org.scalann"

version := "0.1.0-SNAPSHOT"

scalaVersion := "2.10.0"

resolvers ++= Seq(
  "Sonatype repository" at "https://oss.sonatype.org/content/repositories/snapshots/"
)

libraryDependencies ++= Seq(
  "org.scalanlp" % "breeze-math_2.10" % "0.2-SNAPSHOT",
  "org.scalanlp" % "breeze-viz_2.10" % "0.2-SNAPSHOT"
)

libraryDependencies ++= Seq(
  "org.scalatest" % "scalatest_2.10" % "1.9.1" % "test",
  "junit" % "junit" % "4.8.2" % "test"
)
