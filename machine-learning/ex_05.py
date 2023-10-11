import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./datasets/sizeweight.csv')

plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.scatter('size', 'weight', data=df)
plt.xlabel('size')
plt.ylabel('weight')
plt.show()

X = df['size'].values
Y = df['weight'].values

mean_x = np.mean(X)
mean_y = np.mean(Y)
n = len(X)
numer = 0
denom = 0

for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2

b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

print('y =',b1,'* x +', b0)

max_x = np.max(X) + 100
min_x = np.min(X) - 100
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.scatter('size', 'weight', data=df)
plt.xlabel('size')
plt.ylabel('weight')
plt.plot(x, y, color='#52b920')
plt.show()

ss_t = 0
ss_r = 0
for i in range(X.size):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)

print("r2 (manual) =",r2)

X = X.reshape((X.size, 1))
reg = LinearRegression()
reg = reg.fit(X, Y)
Y_pred = reg.predict(X)
r2_score = reg.score(X, Y)
print("r2 (LinearRegression) =",r2_score)




