from apyori import apriori

records = [
    ['p', 'l'],
    ['p', 'f', 'c', 'o'],
    ['l', 'f', 'c', 'r'],
    ['p', 'l', 'f', 'c'],
    ['p', 'l', 'f', 'r']
]

rules = apriori(records, min_support=0.5, min_confidence=0.9)

for item in list(rules):
    pair = item[0]
    items = [x for x in pair]
    ant = str(list(item[2][0][0]))[1:-1]
    cons = str(list(item[2][0][1]))[1:-1]
    print("Rule: {" + ant + "} -> {" + cons + "}")
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")