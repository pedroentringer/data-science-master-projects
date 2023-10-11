import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_csv('./datasets/House.csv')
df_new = pd.read_csv('./datasets/House_Final_new.csv')

X = df[['HouseSize', 'LotSize', 'Bedrooms', 'Granite', 'Bathroom']]
Y = df[['SellingPrice']]

regr_1 = DecisionTreeRegressor()
regr_1.fit(X, Y)

r = export_text(regr_1, feature_names=['HouseSize', 'LotSize', 'Bedrooms', 'Granite', 'Bathroom'])
print(r)

plt.figure()
plot_tree(regr_1)
plt.show()

predictions = regr_1.predict(X=df[['HouseSize', 'LotSize', 'Bedrooms', 'Granite', 'Bathroom']])
mse = mean_squared_error(df[['SellingPrice']], predictions)
print("MSE:", mse)

predictions = regr_1.predict(X=df_new[['HouseSize', 'LotSize', 'Bedrooms', 'Granite', 'Bathroom']])
print("Predicted price: % d\n"% predictions)

