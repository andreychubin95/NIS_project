package hse.nis

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, concat, expr, lit}
import org.apache.spark.sql.types.{DoubleType, IntegerType}

import utils.SparkStart

object Preparation extends App {
  val spark: SparkSession = SparkStart("Filtering")

  val path: String = "/Users/andreychubin/Desktop/ВШЭ/НИС_2/Project/avito-prohibited-content"
  val loadPath: String = path + "/avito_train.tsv"
  val savePath: String = path + "/data"

  val data: DataFrame = spark.read.format("csv")
    .option("sep", "\t")
    .option("inferSchema", "true")
    .option("header", "true")
    .load(loadPath)

  val cleaned_data: DataFrame = data.filter("itemid is not null and is_blocked is not null")
  val dataFrame: DataFrame = cleaned_data.select(
    col("itemid"),
    col("category"),
    concat(
      col("subcategory"),
      lit(" "),
      col("description"),
      lit(" "),
      col("attrs")
    ).alias("text"),
    col("price").cast(DoubleType),
    col("phones_cnt"),
    col("emails_cnt"),
    col("urls_cnt").cast(IntegerType),
    expr("cast(is_blocked as Int) as label")
  )

  val train: DataFrame = dataFrame.filter("cast(itemid as Int) % 3 == 0")
  val test: DataFrame = dataFrame.filter("cast(itemid as Int) % 3 != 0")

  train.write.mode("overwrite").format("parquet").save(savePath + "/train")
  test.write.mode("overwrite").format("parquet").save(savePath + "/test")

  spark.stop()

}
