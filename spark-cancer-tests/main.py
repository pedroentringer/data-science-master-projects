from time import sleep
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('spark').getOrCreate()
data = spark.read.csv('./datasets/boston_housing.csv', header=True, inferSchema=True)
data.show()

def pedro_teste():
    print("pedro")
    sleep(3)


pedro_teste()

feature_columns = data.columns[:-1] # here we omit the final column
assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")
data_2 = assembler.transform(data)

train, test = data_2.randomSplit([0.7, 0.3])
algo = LinearRegression(featuresCol="features", labelCol="medv")
model = algo.fit(train)
evaluation_summary = model.evaluate(test)
print("MAE: ", evaluation_summary.meanAbsoluteError)
print("RMSE: ", evaluation_summary.rootMeanSquaredError)
print("R2: ", evaluation_summary.r2)
sleep(100)
