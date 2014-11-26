common.settings

name := "scalann-examples"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze-math" % common.breezeVersion,
  "org.scalanlp" %% "breeze-viz" % common.breezeVersion,
  "org.scalanlp" %% "breeze-learn" % common.breezeVersion
)

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % common.scalaTestVersion % "test",
  "com.google.caliper" % "caliper" % "0.5-rc1" % "test"
)
