import pyspark.sql.functions as f


from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def create_pipeline(fields):
    assembler = VectorAssembler(inputCols=fields, outputCol="features")
    rf = RandomForestClassifier(
        labelCol="label", featuresCol="features", numTrees=30, maxDepth=10
    )
    return Pipeline(stages=[assembler, rf])


def train_model(fields, training_data):
    pipeline = create_pipeline(fields)
    model = pipeline.fit(training_data)
    return model


def evaluate_model(model, input_data):
    valid_preds = model.transform(input_data)
    acc = BinaryClassificationEvaluator().evaluate(valid_preds)
    valid_preds.show(truncate=False)
    print(acc)
