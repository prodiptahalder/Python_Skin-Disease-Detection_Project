import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def classify():
    data = pd.read_csv("processed_Data_all.csv")
    training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)
    X_train = training_set.iloc[:, 0:-1].values
    Y_train = training_set.iloc[:, -1].values
    X_test = test_set.iloc[:, 0:-1].values
    Y_test = test_set.iloc[:, -1].values

    # Naive Bayes Implementation
    NB_classifier = GaussianNB()
    NB_classifier.fit(X_train, Y_train)
    NB_y_pred = NB_classifier.predict(X_test)
    NB_y_pred_train = NB_classifier.predict(X_train)

    test_set["Naive Bayes Predictions"] = NB_y_pred

    cm1 = confusion_matrix(Y_test, NB_y_pred)
    accuracy = float(cm1.diagonal().sum()) / len(Y_test)
    print("\nAccuracy Of Naive Bayes For The Given Dataset : ", accuracy)

    # outcome values order in sklearn
    tp, fn, fp, tn = confusion_matrix(Y_test, NB_y_pred, labels=[1, 0]).reshape(-1)
    print('Outcome values (tp,fn,fp,tn) : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy
    report_matrix = classification_report(Y_test, NB_y_pred, labels=[1, 0])
    print('Classification report for Naive Bayes classifier : \n', report_matrix)

