import pydotplus
import pandas as pd
from sklearn import tree
from statistics import mean
from IPython.display import Image
from sklearn.tree import export_text
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate

df = pd.read_csv("./datasets/zoo.csv")

X = df[['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']]
Y = df[['type']]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

r = export_text(clf, feature_names=['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize'])
print(r)

dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator','toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize'],
    class_names=clf.classes_
)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_pdf("./results/dt.pdf")

clf = tree.DecisionTreeClassifier()

y_pred = cross_val_predict(clf, X, Y, cv=10)

print(confusion_matrix(Y, y_pred))
print(classification_report(Y, y_pred))

cv_results = cross_validate(clf, X, Y, cv=10)
print("Accuracy:", mean(cv_results['test_score']))