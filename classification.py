from machineLearning_functions import decisionTreeClassifier
from machineLearning_functions import knnClassifier
from machineLearning_functions import logisticRegressionClassifier
from machineLearning_functions import naiveBayesClassifier
from machineLearning_functions import randomForestClassifier
from machineLearning_functions import SVMclassifier

def classify_images():
    decisionTreeClassifier.classify()
    randomForestClassifier.classify()

classify_images()