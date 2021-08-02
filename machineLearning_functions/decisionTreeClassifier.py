import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree

def classify():

    data = pd.read_csv("selected_Feature_Data_all.csv")
    #data = pd.read_csv("selected_Feature_Data_all.csv")
    training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)
    X_train = training_set.iloc[:, 0:-1].values
    Y_train = training_set.iloc[:, -1].values
    X_test = test_set.iloc[:, 0:-1].values
    Y_test = test_set.iloc[:, -1].values

    # Decision Tree Classification implementation
    DT_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    DT_classifier.fit(X_train, Y_train)
    DT_y_pred = DT_classifier.predict(X_test)

    #text_representation = tree.export_text(DT_classifier)
    #print(text_representation)

    test_set["Decision Tree Predictions"] = DT_y_pred

    cm4 = confusion_matrix(Y_test, DT_y_pred)
    accuracy = float(cm4.diagonal().sum()) / len(Y_test)
    print("\nAccuracy Of Decision Tree For The Given Dataset : ", accuracy)

    tp, fn, fp, tn = confusion_matrix(Y_test, DT_y_pred, labels=[0, 1]).reshape(-1)
    print('True Positive : ', tp)
    print('False Negative : ', fn)
    print('False Positive : ', fp)
    print('True Negative : ', tn)

    # classification report for precision, recall f1-score and accuracy
    report_matrix = classification_report(Y_test, DT_y_pred, labels=[0,1])
    print('Classification report of Decision Tree (0->melanoma and 1->non-melanoma) : \n', report_matrix)




