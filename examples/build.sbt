common.settings

name := "scalann-examples"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % common.breezeVersion,
  "org.scalanlp" %% "breeze-viz" % common.breezeVizVersion,
  "org.scalanlp" %% "nak" % common.nakVersion
)

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % common.scalaTestVersion % "test",
  "com.google.caliper" % "caliper" % "0.5-rc1" % "test"
)
