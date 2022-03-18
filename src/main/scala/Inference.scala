package hse.nis

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.PipelineModel
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel

import utils.{SparkStart, asDense, inferencePath, modelSavePath, nlpSavePath, pipelineSavePath, testPath}
import nlp.NLPProcessing


object Inference extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val currentLogger: Logger = Logger.getLogger("his")
  currentLogger.setLevel(Level.INFO)
  currentLogger.info("Job started ...")

  val spark: SparkSession = SparkStart("inference")
  val nlpPipeline: NLPProcessing = new NLPProcessing(nlpSavePath)
  val pipeline: PipelineModel = PipelineModel.load(pipelineSavePath)
  val xgboost: XGBoostClassificationModel = XGBoostClassificationModel.load(modelSavePath)

  val data: DataFrame = spark.read.parquet(testPath).na.fill(0.0)
    .filter("itemid % 156 = 0") // otherwise I get OutOfMemoryError

  val transformed: DataFrame = nlpPipeline.transform(data)
  currentLogger.info("Data tokenized and lemmatized")

  val finalized: DataFrame = pipeline.transform(transformed)
    .withColumn("features", asDense(col("features")))
  currentLogger.info("Data vectorized")

  val predictions: DataFrame = xgboost.transform(finalized)
  currentLogger.info("Inference made")

  predictions.write.mode("overwrite").parquet(inferencePath)
  currentLogger.info("Data written")

  spark.stop()

}
