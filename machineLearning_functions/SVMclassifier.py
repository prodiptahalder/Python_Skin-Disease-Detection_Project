import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def classify():
    data = pd.read_csv("processed_Data_all.csv")
    training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)
    X_train = training_set.iloc[:, 0:-1].values
    Y_train = training_set.iloc[:, -1].values
    X_test = test_set.iloc[:, 0:-1].values
    Y_test = test_set.iloc[:, -1].values

    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)

    '''
    # SVM rbf implementation
    classifier = SVC(kernel='rbf')
    classifier.fit(X_train,Y_train)
    Y_pred=classifier.predict(X_test)

    # SVM linear implementation
    SVC_linear_classifier = SVC(kernel='linear')
    SVC_linear_classifier.fit(X_train,Y_train)
    SVC_linear_Y_pred=SVC_linear_classifier.predict(X_test)
    '''

    # SVM sigmoid implementation
    SVC_sigmoid_classifier = SVC(kernel='sigmoid')
    SVC_sigmoid_classifier.fit(X_train, Y_train)
    SVC_sigmoid_y_pred = SVC_sigmoid_classifier.predict(X_test)
    '''
    # SVM polynomial implementation
    SVC_poly_classifier = SVC(kernel='poly')
    SVC_poly_classifier.fit(X_train,Y_train)
    SVC_poly_Y_pred=SVC_poly_classifier.predict(X_test)
    '''
    # test_set["SVM rbf Predictions"] = Y_pred
    # test_set["SVM linear Predictions"] = SVC_linear_Y_pred
    test_set["SVM sigmoid Predictions"] = SVC_sigmoid_y_pred
    # test_set["SVM poly Predictions"] = SVC_poly_Y_pred

    cm = confusion_matrix(Y_test, SVC_sigmoid_y_pred)
    accuracy = float(cm.diagonal().sum()) / len(Y_test)
    print("\nAccuracy Of SVM sigmoid kernel For The Given Dataset : ", accuracy)

    '''
    cm = confusion_matrix(Y_test,Y_pred)
    accuracy = float(cm.diagonal().sum())/len(Y_test)
    print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)
    '''

    tp, fn, fp, tn = confusion_matrix(Y_test, SVC_sigmoid_y_pred, labels=[0, 1]).reshape(-1)
    print('Outcome values of SVM sigmoid kernel (tp,fn,fp,tn) : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy
    report_matrix = classification_report(Y_test, SVC_sigmoid_y_pred, labels=[0, 1])
    print('Classification report of SVM sigmoid kernel : \n', report_matrix)

