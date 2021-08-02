from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def setData(n):
    return (n + 1)

def my_function():
    data = pd.read_csv("dataset.csv")
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(X)
    normalizer = Normalizer().fit(rescaledX)
    normalizedX = normalizer.transform(rescaledX)
    df = pd.DataFrame(normalizedX)
    Y = y.array
    df['DiseaseType'] = Y
    df.to_csv("processed_Data_all.csv")
    data1 = pd.read_csv("processed_Data_all.csv")
    X1 = data1.iloc[:, 1:-1]
    y1 = data1.iloc[:, -1]
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X1, y1)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    fA = featureScores.nlargest(200, 'Score').index.array
    featureArray = map(setData, fA)
    fA1 = list(featureArray)
    X2 = data1.iloc[:, fA1]
    df1 = pd.DataFrame(X2)
    df1['DiseaseType'] = Y
    df1.to_csv("selected_Feature_Data_all.csv")

print("Calling the function...")
my_function()
print("Called the function...")