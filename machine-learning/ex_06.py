import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

df = pd.read_csv('./datasets/House.csv')
df_new = pd.read_csv('./datasets/House_Final_new.csv')

linear_regressor = LinearRegression()
linear_regressor.fit(df[['HouseSize', 'LotSize', 'Bedrooms', 'Granite', 'Bathroom']], df['SellingPrice'])
"""
SellingPrice = -26.930784 * HouseSize +
 6.334524 * LotSize +
 44293.760584 * Bedrooms +
 7140.676293 * Granite +
 43179.199889 * Bathroom +
 -21739.29666506665
"""

print("coefs:", linear_regressor.coef_, sep=" ")
print("intercept:", linear_regressor.intercept_, sep=" ")

np.set_printoptions(formatter={'float_kind': '{:f}'.format})

linear_regressor = LinearRegression()
linear_regressor.fit(df[['HouseSize', 'LotSize', 'Bedrooms', 'Bathroom']], df['SellingPrice'])

"""
SellingPrice = -26.688240 * HouseSize +
 7.055124 * LotSize +
 43166.076944 * Bedrooms +
 42292.090237 * Bathroom +
 -21661.12129661304
"""

predictions = linear_regressor.predict(df[['HouseSize', 'LotSize', 'Bedrooms','Bathroom']])
print("\n= METRICS =")
print('mean_absolute_error: ', mean_absolute_error(df['SellingPrice'], predictions), sep="")
print('Root mean_squared_error: ', math.sqrt(mean_squared_error(df['SellingPrice'], predictions)), sep="")

predictions2 = linear_regressor.predict(df_new[['HouseSize', 'LotSize', 'Bedrooms', 'Bathroom']])
print("\nmodel final predictions:", predictions2, sep=" ")

