package hse.nis

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import utils.{SparkStart, asDense, dataSavePath, modelSavePath}

object ModelTraining extends App {
  val spark: SparkSession = SparkStart("modelling")

  val df_train: DataFrame = spark.read.parquet(dataSavePath)
  val train: DataFrame = df_train.withColumn("features", asDense(col("features")))

  val classifier: XGBoostClassifier = new XGBoostClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setNumRound(10)
    .setNumWorkers(1)

  val classifierModel: XGBoostClassificationModel = classifier.fit(train)

  classifierModel.write.overwrite().save(modelSavePath)

  spark.stop()

}
