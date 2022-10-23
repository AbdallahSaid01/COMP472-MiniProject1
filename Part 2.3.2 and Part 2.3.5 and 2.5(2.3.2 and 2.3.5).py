import json as js
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

filea = open('goemotions.json')
fileb = js.load(filea)
# Part 2.1
posts = [item[0] for item in fileb]
vec = CountVectorizer()
vec.fit(posts)
split = np.array(fileb)
x = split[:, 0]
y1 = split[:, 1]
y2 = split[:, 2]
#Part 2.2
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, train_size=0.8, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, train_size=0.8, test_size=0.2)
# Part 2.3.2
x1_train = vec.fit_transform(x1_train)
x1_test = vec.transform(x1_test)
classifier1 = DecisionTreeClassifier()
classifier1.fit(x1_train, y1_train)
y1_pred = classifier1.predict(x1_test)

print("For predicting emotions : \n")
print(classification_report(y1_test, y1_pred))
print("The Confusion Matrix of Decision Tree(Emotions): \n", confusion_matrix(y1_test, y1_pred))

x2_train = vec.fit_transform(x2_train)
x2_test = vec.transform(x2_test)
classifier2 = DecisionTreeClassifier()
classifier2.fit(x2_train, y2_train)
y2_pred = classifier2.predict(x2_test)

print("For predicting sentiments : \n")
print(classification_report(y2_test, y2_pred))
print("The Confusion Matrix of Decision Tree(Sentiments): \n", confusion_matrix(y2_test, y2_pred))

pd={"criterion":["entropy"], "max_depth": [2,11],"min_samples_split": [12,40,6]}
grida=GridSearchCV(estimator=classifier1,param_grid=pd,n_jobs=-1,error_score='raise')
grida.fit(x1_train, y1_train)
y1_pred = grida.predict(x1_test)

print("For predicting emotions using GridSearchCV: \n")
print(classification_report(y1_test, y1_pred))
print("The Confusion Matrix of Decision Tree(Emotions) Using GridSearchCV: \n", confusion_matrix(y1_test, y1_pred))
print("\n The best estimator across ALL searched params:\n", grida.best_estimator_)
print("\n The best score across ALL searched params:\n", grida.best_score_)
print("\n The best parameters across ALL searched params:\n", grida.best_params_)

classifier2 = DecisionTreeClassifier()
gridb=GridSearchCV(estimator=classifier2,param_grid=pd,n_jobs=-1,error_score='raise')
gridb.fit(x2_train, y2_train)
y2_pred = gridb.predict(x2_test)

print("For predicting sentiments using GridSearchCV: \n")
print(classification_report(y2_test, y2_pred))
print("The Confusion Matrix of Decision Tree(Sentiments) Using GridSearchCV: \n", confusion_matrix(y2_test, y2_pred))
print("\n The best estimator across ALL searched params:\n", gridb.best_estimator_)
print("\n The best score across ALL searched params:\n", gridb.best_score_)
print("\n The best parameters across ALL searched params:\n", gridb.best_params_)
filea.close()
