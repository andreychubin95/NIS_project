package hse.nis

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

package object utils {
  val dataPath: String = "/Users/andreychubin/Desktop/ВШЭ/НИС_2/Project/avito-prohibited-content/data/train"
  val testPath: String = "/Users/andreychubin/Desktop/ВШЭ/НИС_2/Project/avito-prohibited-content/data/test"
  val dataSavePath: String = "/Users/andreychubin/Desktop/ВШЭ/НИС_2/Project/avito-prohibited-content/data/ml_ready_train"
  val nlpSavePath: String = "/Users/andreychubin/Desktop/ВШЭ/НИС_2/Project/models/nlpPipeline"
  val pipelineSavePath: String = "/Users/andreychubin/Desktop/ВШЭ/НИС_2/Project/models/generalPipeline"
  val featuresColumns: Array[String] = Array("normalized", "price", "phones_cnt", "emails_cnt", "urls_cnt")
  val modelSavePath: String = "/Users/andreychubin/Desktop/ВШЭ/НИС_2/Project/models/xgboost"
  val inferencePath: String = "/Users/andreychubin/Desktop/ВШЭ/НИС_2/Project/avito-prohibited-content/data/inference"
  val asDense: UserDefinedFunction = udf((v: Vector) => v.toDense)
}
