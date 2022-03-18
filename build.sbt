name := "NIS"

version := "1.0"

scalaVersion := "2.12.10"

idePackagePrefix := Some("hse.nis")

libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp-spark32" % "3.4.2"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.2.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.2.1"
libraryDependencies += "ml.dmlc" %% "xgboost4j" % "1.5.2"
libraryDependencies += "ml.dmlc" %% "xgboost4j-spark" % "1.5.2"
