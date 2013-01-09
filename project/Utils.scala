package scalann.sbtplugins

import sbt._

import Defaults._
import Keys._

/**
 * Extracted from https://github.com/chris-twiner/scalesXml/blob/master/project/Utils.scala
 */
object Utils {

  /**
   * At this late stage the properties can be used
   */
  def cpRunnerInit(config: sbt.Configuration): Project.Initialize[Task[ScalaRun]] =
    (taskTemporaryDirectory, scalaInstance, baseDirectory, javaOptions, outputStrategy, javaHome, trapExit, fullClasspath in config) map { (tmp, si, base, options, strategy, javaHomeDir, trap, cpa) =>
        val cp = "-classpath" :: Path.makeString(cpa.files) :: Nil

        new ForkRun(
          ForkOptions(scalaJars = si.jars, javaHome = javaHomeDir, outputStrategy = strategy,
            runJVMOptions = options ++ cp,
            workingDirectory = Some(base)))

    }

  /**
   * Runner that can run caliper tasks or any others requiring the classpath be specified on the forked app
   */
  def caliperRunTask(scoped: ScopedTask[Unit], config: sbt.Configuration, arguments: String*): Setting[Task[Unit]] =
    scoped <<= (initScoped(scoped.scopedKey, cpRunnerInit(config)) zipWith (fullClasspath in config, streams).identityMap) {
      case (rTask, t) =>
        (t :^: rTask :^: KNil) map {
          case (cp, s) :+: r :+: HNil =>
            sbt.toError(r.run("com.google.caliper.Runner", Build.data(cp),
              arguments, s.log))
        }
    }

}