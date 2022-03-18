package hse.nis

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, IDF, VectorAssembler}

import nlp.NLPProcessing
import utils.{SparkStart, dataPath, nlpSavePath, featuresColumns, pipelineSavePath, dataSavePath}


object TrainPreparation extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val currentLogger: Logger = Logger.getLogger("his")
  currentLogger.setLevel(Level.INFO)
  currentLogger.info("Job started ...")

  val spark: SparkSession = SparkStart("test")
  currentLogger.info("Spark Session started ...")

  val data = spark.read.parquet(dataPath)
    .na.fill(0.0)
    .filter("itemid % 221 == 0")

  currentLogger.info(s"Data size appx.: ${data.sample(0.1).count() * 100} rows") // makes count much faster
  currentLogger.info("Data read")

  val nlp: NLPProcessing = new NLPProcessing(nlpSavePath)
  val transformed: DataFrame = nlp.fitAndTransform(data)
  currentLogger.info("Data went trough NLP processing")

  val hashingTF = new HashingTF()
    .setNumFeatures(10000)
    .setInputCol("tokens")
    .setOutputCol("rawFeatures")

  val idf = new IDF()
    .setMinDocFreq(20)
    .setInputCol("rawFeatures")
    .setOutputCol("normalized")

  val vectorAssembler: VectorAssembler = new VectorAssembler()
    .setInputCols(featuresColumns)
    .setOutputCol("features")

  val featuresPipeline: PipelineModel = new Pipeline()
    .setStages(Array(hashingTF, idf, vectorAssembler))
    .fit(transformed)

  featuresPipeline.write.overwrite().save(pipelineSavePath)

  println(transformed.columns.mkString(", "))

  currentLogger.info("Feature Pipeline build")

  val transformedData: DataFrame = featuresPipeline
    .transform(transformed)
    .select("itemid", "features", "label")

  currentLogger.info("Data transformed")

  println(transformedData.show(5))

  currentLogger.info("Writing data ...")
  transformedData.write.mode("overwrite").format("parquet").save(dataSavePath)
  currentLogger.info("Data written")
  currentLogger.info("Done")

  spark.stop()

}
