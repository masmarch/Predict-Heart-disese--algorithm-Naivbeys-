import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
import sklearn.naive_bayes
import pandas as pd


# prepare data
filename = "CS_data.csv"
splitRatio = 0.80
dataset = pd.read_csv(filename).values
trainingSet, testSet = train_test_split(dataset, train_size=splitRatio)

# split input/output
# X: (216, 13), y: (216, 1)
X_train = trainingSet[:, :13]
X_test = testSet[:, :13]
Y_train = trainingSet[:, 13] - 1
Y_test = testSet[:, 13] - 1


# train model NaiveBayes
nb = sk.naive_bayes.GaussianNB()
nb.fit(X_train, Y_train)
nb_test_score = round(nb.score(X_test, Y_test) * 100, 2)
print("naivebayes test score = ", nb_test_score, "%")

