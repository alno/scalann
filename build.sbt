common.settings

name := "scalann"

// Subprojects
val core = project in file("core")
val examples = project in file("examples") dependsOn core
