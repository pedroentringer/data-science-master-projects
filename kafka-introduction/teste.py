import pandas as pd
from pyspark.shell import spark

train_data_no_name = pd.read_csv("./datasets/train_FD001.txt", sep="\s+", header=None)
columns = ['engineNumber', 'cycleNumber', 'opSetting1',
           'opSetting2', 'opSetting3', 'sensor1', 'sensor2',
           'sensor3', 'sensor4', 'sensor5', 'sensor6',
           'sensor7', 'sensor8', 'sensor9', 'sensor10',
           'sensor11', 'sensor12', 'sensor13', 'sensor14',
           'sensor15', 'sensor16',
           'sensor17', 'sensor18', 'sensor19', 'sensor20',
           'sensor21']

df = spark.createDataFrame(train_data_no_name, columns)
print(df.corr('engineNumber', 'sensor1'))
