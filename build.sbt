
name := "scalann"
organization := "org.scalann"
version := "0.1.0-SNAPSHOT"

scalaVersion := "2.10.1"

// Subprojects
val core = project in file("core")
val examples = project in file("examples") dependsOn core
