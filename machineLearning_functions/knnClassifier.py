import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def classify():
    data = pd.read_csv("processed_Data_all.csv")
    training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)
    X_train = training_set.iloc[:, 0:-1].values
    Y_train = training_set.iloc[:, -1].values
    X_test = test_set.iloc[:, 0:-1].values
    Y_test = test_set.iloc[:, -1].values

    # K-NN implementation
    KNN_classifier = KNeighborsClassifier(n_neighbors=30, metric='minkowski', p=2)
    KNN_classifier.fit(X_train, Y_train)
    KNN_y_pred = KNN_classifier.predict(X_test)

    test_set["KNN Predictions"] = KNN_y_pred

    cm3 = confusion_matrix(Y_test, KNN_y_pred)
    accuracy = float(cm3.diagonal().sum()) / len(Y_test)
    print("\nAccuracy Of KNN For The Given Dataset : ", accuracy)

    tp, fn, fp, tn = confusion_matrix(Y_test, KNN_y_pred, labels=[0, 1]).reshape(-1)
    print('Outcome values of KNN Classifier (tp,fn,fp,tn) : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy
    report_matrix = classification_report(Y_test, KNN_y_pred, labels=[0, 1])
    print('Classification report of KNN Classifier : \n', report_matrix)
