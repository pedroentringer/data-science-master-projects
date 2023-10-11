import pandas as pd

df = pd.read_csv("./datasets/dataset.csv")

print(df.shape)
# (5842, 2)
# (Number of Rows, Number of columns)

print(df.columns)
# Column name

exit(0)
#Close without error