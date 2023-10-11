from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.shell import spark

# create a sample dataframe with 4 features and 1 label column
sample_data_train = spark.createDataFrame([
    (2.0, 'A', 'S10', 40, 1.0),
    (1.0, 'X', 'E10', 25, 1.0),
    (4.0, 'X', 'S20', 10, 0.0),
    (3.0, 'Z', 'S10', 20, 0.0),
    (4.0, 'A', 'E10', 30, 1.0),
    (2.0, 'Z', 'S10', 40, 0.0),
    (5.0, 'X', 'D10', 10, 1.0),
], ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'label'])

# view the data
sample_data_train.show()

# define stage 1: transform the column feature_2 to numeric
stage_1 = StringIndexer(inputCol= 'feature_2', outputCol= 'feature_2_index')
# define stage 2: transform the column feature_3 to numeric
stage_2 = StringIndexer(inputCol= 'feature_3', outputCol= 'feature_3_index')

# define stage 3: create a vector of all the features required to train the logistic regression model
stage_3 = VectorAssembler(inputCols=['feature_1', 'feature_2_index', 'feature_3_index', 'feature_4'],
                          outputCol='features')

# define stage 4: logistic regression model
stage_4 = LogisticRegression(featuresCol='features',labelCol='label')

paramGrid = ParamGridBuilder().addGrid (stage_4.regParam, [0.1 ,
0.01]).build()

tvs = TrainValidationSplit ( estimator = stage_4 ,
estimatorParamMaps = paramGrid ,
evaluator = BinaryClassificationEvaluator() ,
trainRatio =0.8)


# setup the pipeline
regression_pipeline = Pipeline(stages= [stage_1, stage_2, stage_3, tvs])

# fit the pipeline for the trainind data
model = regression_pipeline.fit(sample_data_train)
# transform the data
sample_data_train = model.transform(sample_data_train)
sample_data_train.show()
# view some of the columns generated
sample_data_train.select('features', 'label', 'prediction').show()

# create a sample data without the labels
sample_data_test = spark.createDataFrame([
    (3.0, 'Z', 'S10', 40),
    (1.0, 'X', 'E10', 20),
    (4.0, 'A', 'S20', 10),
    (3.0, 'A', 'S10', 20),
    (4.0, 'X', 'D10', 30),
    (1.0, 'Z', 'E10', 20),
    (4.0, 'A', 'S10', 30),
], ['feature_1', 'feature_2', 'feature_3', 'feature_4'])

# transform the data using the pipeline
sample_data_test = model.transform(sample_data_test)

# see the prediction on the test data
sample_data_test.select('features', 'prediction').show()