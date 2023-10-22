# installed packages
from pyspark.sql import SparkSession
from graphframes import *
from data_loader import friends

spark = (
    SparkSession.builder.config("spark.driver.memory", "6g")
    .config("spark.executor.memory", "6g")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("FATAL")

# create example graph dataframe
# g = friends(spark)
print("SUCCESS!")
spark.stop()
