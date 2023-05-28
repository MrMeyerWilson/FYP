import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

data = pd.read_csv('features.csv')

X = data[data.columns[1:-1]] 

y = data[data.columns[-1]]

X,y = X.to_numpy(), y.to_numpy()
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)


classifier = KNeighborsClassifier(n_neighbors = 9)

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions) 

recall = recall_score(y_test, predictions) 
f1 = f1_score(y_test, predictions) 

print("\n")
print(accuracy)
print(recall)
print(f1)

print(classification_report(y_test, predictions)) 
