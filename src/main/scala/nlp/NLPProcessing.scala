package hse.nis
package nlp

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, expr, udf}

class NLPProcessing(path: String) {
  private def getResult(dataFrame: DataFrame, pipelineModel: PipelineModel): DataFrame = {
    val asFilteredList = udf((x: Array[String]) => x.filter((word: String) => word.length > 2))
    val tokenized = pipelineModel.transform(dataFrame)
    tokenized.withColumn("tokens", expr("lemma.result"))
      .withColumn("tokens", asFilteredList(col("tokens")))
  }

  def fitAndTransform(dataFrame: DataFrame, savePath: String = this.path): DataFrame = {
    val pipeline = BuildPipeline(dataFrame).pipeline
    pipeline.write.overwrite().save(savePath)
    getResult(dataFrame, pipeline)
  }

  def transform(dataFrame: DataFrame, savePath: String = this.path): DataFrame = {
    val pipeline: PipelineModel = PipelineModel.load(savePath)
    getResult(dataFrame, pipeline)
  }
}
