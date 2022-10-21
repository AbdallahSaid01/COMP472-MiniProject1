import json as js
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.metrics import classification_report, confusion_matrix

f = open('goemotions.json')
file = js.load(f)

# Part 2.1
sentences = [post[0] for post in file]
vec = CountVectorizer()
vec.fit(sentences)
split = np.array(file)
x = split[:, 0]
y1 = split[:, 1]
y2 = split[:, 2]

# Part 2.2
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, train_size=0.8, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, train_size=0.8, test_size=0.2)

#Part 2.3.3
perceptron_classifier = MLPClassifier(max_iter=5)
print("We are using a Multi-Layered Perception Classifier or MLPClassifier with the default parameters.")
x1_train = vec.fit_transform(x1_train)
x1_test = vec.transform(x1_test)
perceptron_classifier.fit(x1_train, y1_train)
y1_pred = perceptron_classifier.predict(x1_test)
print("Emotions: \n")
print(classification_report(y1_test, y1_pred))
print("Confusion Matrix: \n", confusion_matrix(y1_test, y1_pred))
x2_train = vec.fit_transform(x2_train)
x2_test = vec.transform(x2_test)
perceptron_classifier = MLPClassifier(max_iter=20)
perceptron_classifier.fit(x2_train, y2_train)
y2_pred = perceptron_classifier.predict(x2_test)
print("Sentiments: \n")
print(classification_report(y2_test, y2_pred))
print("Confusion Matrix: \n", confusion_matrix(y2_test, y2_pred))

f.close()