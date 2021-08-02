import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def classify():
    data = pd.read_csv("processed_Data_all.csv")
    training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)
    X_train = training_set.iloc[:, 0:-1].values
    Y_train = training_set.iloc[:, -1].values
    X_test = test_set.iloc[:, 0:-1].values
    Y_test = test_set.iloc[:, -1].values

    # Logistic Regression Implementation
    LR_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_classifier.fit(X_train, Y_train)
    LR_y_pred = LR_classifier.predict(X_test)

    test_set["Logistic Regressions Predictions"] = LR_y_pred
    cm2 = confusion_matrix(Y_test, LR_y_pred)
    accuracy = float(cm2.diagonal().sum()) / len(Y_test)
    print("\nAccuracy Of Logistic Regression For The Given Dataset : ", accuracy)

    tp, fn, fp, tn = confusion_matrix(Y_test, LR_y_pred, labels=[0, 1]).reshape(-1)
    print('Outcome values of Logistic Regression Classifier(tp,fn,fp,tn) : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy
    report_matrix = classification_report(Y_test, LR_y_pred, labels=[0, 1])
    print('Classification report of Logistic Regression Classifier : \n', report_matrix)

