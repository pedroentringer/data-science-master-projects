import pandas as pd
from apyori import apriori

df = pd.read_csv("./datasets/supermarket.csv")

records = []

for index, row in df.iterrows():
  record = []
  for c in df.columns:
    if row[c] == 't':
     record.append(c)
    if c == "total":
     record.append("total_"+row[c])
  records.append(record)

rules = apriori(records, min_support=0.1, min_confidence=0.9, min_length=2, min_lift=1.25)
listrules = list(rules)

for item in listrules:
   print(item)

for item in listrules:
  pair = item[0]
  items = [x for x in pair]
  ant = str(list(item[2][0][0]))[1:-1]
  cons = str(list(item[2][0][1]))[1:-1]
  print("Rule: {" + ant + "} -> {" + cons + "}")
  print("Support: " + str(item[1]))
  print("Confidence: " + str(item[2][0][2]))
  print("Lift: " + str(item[2][0][3]))
  print("=====================================")