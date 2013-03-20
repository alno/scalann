import sbt._
import Keys._

object ScalannBuild extends Build {

  lazy val root = Project(id = "scalann", base = file(".")) aggregate(core, examples)

  lazy val core = Project(id = "scalann-core", base = file("core"))

  lazy val examples = Project(id = "scalann-examples", base = file("examples")) dependsOn(core)

}
