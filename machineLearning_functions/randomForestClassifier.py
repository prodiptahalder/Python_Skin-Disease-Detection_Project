import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def classify():
    data = pd.read_csv("selected_Feature_Data_all.csv")
    training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)
    X_train = training_set.iloc[:, 0:-1].values
    Y_train = training_set.iloc[:, -1].values
    X_test = test_set.iloc[:, 0:-1].values
    Y_test = test_set.iloc[:, -1].values

    # Random forest Implementation
    RF_classifier = RandomForestClassifier(n_estimators=94)
    RF_classifier.fit(X_train, Y_train)
    RF_y_pred = RF_classifier.predict(X_test)

    test_set["Random Forest Predictions"] = RF_y_pred

    cm5 = confusion_matrix(Y_test, RF_y_pred)
    accuracy = float(cm5.diagonal().sum()) / len(Y_test)
    print("\nAccuracy Of Random Forest For The Given Dataset : ", accuracy)

    tp, fn, fp, tn = confusion_matrix(Y_test, RF_y_pred, labels=[0, 1]).reshape(-1)
    print('True Positive : ', tp)
    print('False Negative : ', fn)
    print('False Positive : ', fp)
    print('True Negative : ', tn)


    # classification report for precision, recall f1-score and accuracy
    report_matrix = classification_report(Y_test, RF_y_pred, labels=[0,1])
    print('Classification report of Random Forest Classifier (0->melanoma and 1->non-melanoma) : \n', report_matrix)

