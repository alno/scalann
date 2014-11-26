name := "scalann-examples"
version := "0.1.0-SNAPSHOT"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze-math" % common.breezeVersion,
  "org.scalanlp" %% "breeze-viz" % common.breezeVersion,
  "org.scalanlp" %% "breeze-learn" % common.breezeVersion
)

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "1.9.1" % "test",
  "com.google.caliper" % "caliper" % "0.5-rc1" % "test",
  "junit" % "junit" % "4.8.2" % "test"
)
