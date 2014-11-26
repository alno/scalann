common.settings

name := "scalann-core"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze-math" % common.breezeVersion,
  "org.scalanlp" %% "breeze-viz" % common.breezeVersion
)

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % common.scalaTestVersion % "test"
)
