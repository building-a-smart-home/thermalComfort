from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("Thermal-Comfort").getOrCreate()

model = PipelineModel.load("./model/thermal-comfort-model") 

kitchen_temp_data = spark.read.option("delimiter", "\t").csv("./data/test-data/Kitchen_Temperature.csv", header=False, inferSchema=True).toDF("timestamp", "ta")
kitchen_humidity_data = spark.read.option("delimiter", "\t").csv("./data/test-data/Kitchen_Humidity.csv", header=False, inferSchema=True).toDF("timestamp", "rh")
kitchen_out_temp_data = spark.read.option("delimiter", "\t").csv("./data/test-data/Kitchen_Virtual_OutdoorTemperature.csv", header=False, inferSchema=True).toDF("timestamp", "t_out")

merged_data = kitchen_temp_data.join(kitchen_humidity_data, "timestamp", "inner").join(kitchen_out_temp_data, "timestamp", "inner").select("timestamp", "ta", "rh", "t_out")

merged_data = merged_data.withColumn("ta_rh_interaction", col("ta") * col("rh"))
merged_data = merged_data.withColumn("timestamp", F.from_unixtime("timestamp"))

predictions = model.transform(merged_data)
predictions.show()

index_model = model.stages[0]  # StringIndexer is typically the first stage
labels = index_model.labels  # Extracting the labels from StringIndexer

# Add a new column with the label for predicted thermal comfort
predictions_with_labels = predictions.withColumn(
    "predicted_thermal_comfort",
    when(predictions["prediction"] == 0, lit(labels[0]))
    .when(predictions["prediction"] == 1, lit(labels[1]))
    .when(predictions["prediction"] == 2, lit(labels[2]))
    .otherwise(lit("Unknown"))
)

# Show the final predictions with labels
predictions_with_labels.select("timestamp", "ta", "rh", "ta_rh_interaction", "prediction", "predicted_thermal_comfort").show()

predictions.select("timestamp", "ta", "rh", "ta_rh_interaction", "prediction").show()

# predictions.select("timestamp", "ta", "rh", "ta_rh_interaction", "prediction") \
#     .write.csv("./data/kitchen_predictions.csv", header=True)

spark.stop()