import json as js
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.metrics import classification_report, confusion_matrix
filea = open('goemotions.json')
fileb = js.load(filea)
#Part 2.1
sentences = [item[0] for item in fileb]
vec = CountVectorizer()
vec.fit(sentences)
split = np.array(fileb)
x = split[:, 0]
y1 = split[:, 1]
y2 = split[:, 2]
# Part 2.2
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, train_size=0.8, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, train_size=0.8, test_size=0.2)

# Part 2.3.1
x_train = vec.fit_transform(x1_train)
x_test = vec.transform(x1_test)

classifier = MultinomialNB()
classifier.fit(x_train, y1_train)

y_pred = classifier.predict(x_test)
print("For predicting emotions : \n")
print(classification_report(y1_test, y_pred))
print("The resulting Confusion Matrix: \n", confusion_matrix(y1_test, y_pred))

x_train = vec.fit_transform(x2_train)
x_test = vec.transform(x2_test)

classifier2 = MultinomialNB()
classifier2.fit(x_train, y2_train)

y_pred = classifier2.predict(x_test)
print("For predicting sentiments : \n")
print(classification_report(y2_test, y_pred))
print("The resulting Confusion Matrix: \n", confusion_matrix(y2_test, y_pred))

# Part 2.3.2
x_train = vec.fit_transform(x1_train)
x_test = vec.transform(x1_test)

classifier = MultinomialNB()
param_dict = {"alpha":[0.5,0,1,2]}
grida= GridSearchCV(estimator=classifier, param_grid=param_dict, n_jobs=-1)
grida.fit(x_train, y1_train)

y_pred = grida.predict(x_test)
print("For predicting emotions using GridSearch: \n")
print(classification_report(y1_test, y_pred))
print("The resulting Confusion Matrix: \n", confusion_matrix(y1_test, y_pred))

x_train = vec.fit_transform(x2_train)
x_test = vec.transform(x2_test)

classifier2 = MultinomialNB()
param_dict = {"alpha":[0.5,0,1,2]}
gridb= GridSearchCV(estimator=classifier2, param_grid=param_dict, n_jobs=-1)
gridb.fit(x_train, y2_train)

y_pred = gridb.predict(x_test)

print("For predicting sentiments GridSearch : \n")
print(classification_report(y2_test, y_pred))
print("The resulting Confusion Matrix: \n", confusion_matrix(y2_test, y_pred))

#2.5(3.1 and 3.4)
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, train_size=0.5, test_size=0.5)
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, train_size=0.5, test_size=0.5)

#2.3.1 in 2.5
x_train = vec.fit_transform(x1_train)
x_test = vec.transform(x1_test)

classifier = MultinomialNB()
classifier.fit(x_train, y1_train)

y_pred = classifier.predict(x_test)
print("For predicting emotions : \n")
print(classification_report(y1_test, y_pred))
print("The resulting Confusion Matrix: \n", confusion_matrix(y1_test, y_pred))

x_train = vec.fit_transform(x2_train)
x_test = vec.transform(x2_test)

classifier2 = MultinomialNB()
classifier2.fit(x_train, y2_train)

y_pred = classifier2.predict(x_test)
print("For predicting sentiments : \n")
print(classification_report(y2_test, y_pred))
print("The resulting Confusion Matrix: \n", confusion_matrix(y2_test, y_pred))

# Part 2.3.4 in 2.5
x_train = vec.fit_transform(x1_train)
x_test = vec.transform(x1_test)

classifier = MultinomialNB()
param_dict = {"alpha":[0.5,0,1,2]}
grida= GridSearchCV(estimator=classifier, param_grid=param_dict, n_jobs=-1)
grida.fit(x_train, y1_train)

y_pred = grida.predict(x_test)
print("For predicting emotions using GridSearch: \n")
print(classification_report(y1_test, y_pred))
print("The resulting Confusion Matrix: \n", confusion_matrix(y1_test, y_pred))

x_train = vec.fit_transform(x2_train)
x_test = vec.transform(x2_test)

classifier2 = MultinomialNB()
param_dict = {"alpha":[0.5,0,1,2]}
gridb= GridSearchCV(estimator=classifier2, param_grid=param_dict, n_jobs=-1)
gridb.fit(x_train, y2_train)

y_pred = gridb.predict(x_test)

print("For predicting sentiments GridSearch : \n")
print(classification_report(y2_test, y_pred))
print("The resulting Confusion Matrix: \n", confusion_matrix(y2_test, y_pred))



