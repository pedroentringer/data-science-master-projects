from time import sleep

from pyspark.shell import spark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
sc=spark.sparkContext

stringJSONRDD = sc.parallelize(("""
{ "id": "123",
"name": "Katie",
"age": 19,
"eyeColor": "brown"
}""",
"""{
"id": "234",
"name": "Michael",
"age": 22,
"eyeColor": "green"
}""",
"""{
"id": "345",
"name": "Simone",
"age": 23,
"eyeColor": "blue"
}""")
)

swimmersJSON = spark.read.json(stringJSONRDD)

swimmersJSON.createOrReplaceTempView("swimmersJSON")

spark.sql("select * from swimmersJSON").collect()

sleep(1000000)