from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

def main():
    data = pd.read_csv('features.csv')

    X = data[data.columns[0:-1]] 
    y = data[data.columns[-1]]
    X,y = X.to_numpy(), y.to_numpy()
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.20)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size = 0.25)
  
    classifier = KNeighborsClassifier(n_neighbors = 9)

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_val)
    predictions_proba = classifier.predict_proba(X_val)

    accuracy = accuracy_score(y_val, predictions) 
    recall = recall_score(y_val, predictions) 
    f1 = f1_score(y_val, predictions) 

    print("\n")
    print(accuracy)
    print(recall)
    print(f1)
    
    predicted_label = np.where(predictions_proba[:,0] > 0.5, 0, 1)
    conf_matrix = confusion_matrix(y_test, predicted_label, labels = classifier.classes_, normalize = "true")

    display_conf = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = classifier.classes_)
    display_conf.plot()
    plt.show()
    
    print(classification_report(y_val, predictions))
    
    model2 = pickle.load(open("Final_Model5", "rb"))
    print(model2.score(X_test, y_test)) 
    


if __name__ == "__main__":
    main()