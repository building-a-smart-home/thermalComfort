from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
import os
import shutil

#from pyspark.sql import functions as F

spark = SparkSession.builder.appName("Thermal-Comfort-model").config("spark.executor.memory", "4g").config("spark.driver.memory", "4g").config("spark.executor.memoryOverhead", "1g").config("spark.memory.fraction", "0.8").getOrCreate()

#dataframe
data = spark.read.csv("./data/train-data/cleanDataNew.csv", header=True, inferSchema=True)
# data.printSchema()
# data.show(5)
data = data.filter((col("clo") < 0.6) & (col("region") == "asia") & (col("climate") == "A"))#filtering summer clothing and asia and Tropical climate

# to check class balance
data_pd = data.toPandas()
class_counts = data_pd['thermal_comfort'].value_counts()
print(class_counts)

data = data.withColumn("ta_rh_interaction", col("ta") * col("rh"))

#convert from string to num
index = StringIndexer(inputCol="thermal_comfort", outputCol="thermal_comfort_index")
#index_model = index.fit(data)

#indexed_data = index.fit(data).transform(data)
#indexed_data.select("thermal_comfort", "thermal_comfort_index").distinct().orderBy("thermal_comfort_index").show()
print("Index labels: ", index.fit(data).labels)

assembler = VectorAssembler().setInputCols(["ta", "rh", "t_out", "ta_rh_interaction"]).setOutputCol("features") #set label using air temperature, humidty, outside temperature
#, "clo", "vel", "met" excluded to apply for temp and humidity

#preprocessing to prevent features with larger numerical ranges
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

#decision tree model setup
setUp = RandomForestClassifier(featuresCol="scaled_features", labelCol="thermal_comfort_index")

paramGrid = ParamGridBuilder() \
    .addGrid(setUp.numTrees, [100, 200, 300]) \
    .addGrid(setUp.maxDepth, [5, 10, 15]) \
    .addGrid(setUp.maxBins, [32, 64, 128]) \
    .addGrid(setUp.subsamplingRate, [0.7, 0.8, 0.9]) \
    .build()

# Cross-validator setup
evaluator = MulticlassClassificationEvaluator(labelCol="thermal_comfort_index", predictionCol="prediction", metricName="accuracy")
crossval = CrossValidator(estimator=setUp, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

#chain data processing and modelling
pipeline = Pipeline(stages=[index, assembler, scaler, crossval])

#split data for training with random consistent splitting with seed for multiple run
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

#model training
model = pipeline.fit(train_data)

model_path = './model/thermal-comfort-model'
if os.path.exists(model_path):
    shutil.rmtree(model_path)
model.save(model_path)

#test model
predict = model.transform(test_data)
# predict.select("thermal_comfort", "prediction").show(5)

accuracy = evaluator.evaluate(predict)
evaluator_precision = MulticlassClassificationEvaluator(labelCol="thermal_comfort_index", predictionCol="prediction", metricName="precision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="thermal_comfort_index", predictionCol="prediction", metricName="recall")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="thermal_comfort_index", predictionCol="prediction", metricName="f1")

precision = evaluator_precision.evaluate(predict)
recall = evaluator_recall.evaluate(predict)
f1 = evaluator_f1.evaluate(predict)

print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

# #evaluate
# evaluator = MulticlassClassificationEvaluator(labelCol="thermal_comfort_index", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator.evaluate(predict)
# print(f"Accuracy: {accuracy}")

spark.stop()