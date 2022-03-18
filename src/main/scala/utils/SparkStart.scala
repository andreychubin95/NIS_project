package hse.nis
package utils

import org.apache.spark.sql.SparkSession

object SparkStart {
  def apply(appName: String, master: String = "local"): SparkSession = {
    val spark: SparkSession = SparkSession.builder
      .appName(appName)
      .config("spark.master", master)
      .getOrCreate()

    spark
  }
}
