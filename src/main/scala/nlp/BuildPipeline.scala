package hse.nis
package nlp

import com.johnsnowlabs.nlp.annotator.{LemmatizerModel, TokenizerModel}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.{DocumentAssembler, HasOutputAnnotationCol, HasOutputAnnotatorType}
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.sql.DataFrame

// this class may have problems in distributed mode
class BuildPipeline(dataFrame: DataFrame) {
  assert(this.dataFrame.columns.contains("text"), "Column text not in inputCols")

  final val pipeline: PipelineModel = this.buildPipeline(this.dataFrame)

  private def arrange(dataFrame: DataFrame):
  Array[
    Transformer with DefaultParamsWritable with HasOutputAnnotatorType with HasOutputAnnotationCol
  ] = {
    val documentAssembler: DocumentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer: TokenizerModel = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .fit(dataFrame)

    val lemmatizer: LemmatizerModel = LemmatizerModel.pretrained("lemma", "ru")
      .setInputCols(Array("token"))
      .setOutputCol("lemma")

    Array(documentAssembler, tokenizer, lemmatizer)
  }

  private def buildPipeline(dataFrame: DataFrame): PipelineModel =
    new Pipeline().setStages(this.arrange(dataFrame)).fit(dataFrame)

}

object BuildPipeline {
  def apply(dataFrame: DataFrame): BuildPipeline = new BuildPipeline(dataFrame)
}