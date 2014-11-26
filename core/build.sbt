common.settings

name := "scalann-core"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % common.breezeVersion,
  "org.scalanlp" %% "breeze-viz" % common.breezeVizVersion
)

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % common.scalaTestVersion % "test"
)
