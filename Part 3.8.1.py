from gensim.models import Word2Vec
import json as js
import numpy as np
import gensim.downloader as api
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

model = api.load('glove-twitter-200')

print("loading GoEmotions")
f = open('goemotions.json')
file = js.load(f)
print("GoEmotions Loaded")
split = np.array(file)
x = split[:, 0]
y1 = split[:, 1]
y2 = split[:, 2]
print("tokenizing sentences")
wx = [word_tokenize(sentence) for sentence in x]
print("sentences tokenized")

print("Embedding each sentence as average weight of its words")
sentenceEmbed = []
totalWordsFound = 0
totalWordsNotFound = 0
for i in range(len(wx)):
    wordsFoundInSentence = 0
    wordEmbed = 0.0
    for word in wx[i]:
        if word in model:
            wordEmbed += model[word]
            totalWordsFound += 1
            wordsFoundInSentence += 1
        else:
            totalWordsNotFound += 1
    if wordsFoundInSentence == 0:
        sentenceEmbed.append(np.zeros(len(model['hi'])))
    else:
        sentenceEmbed.append(wordEmbed/wordsFoundInSentence)


print("Embedding done and printing array")
print("Calculating hit and miss rates")
hitRate = (totalWordsFound/(totalWordsFound+totalWordsNotFound))*100
missRate = (totalWordsNotFound/(totalWordsFound+totalWordsNotFound))*100

print('hitrate = ', hitRate)

x1_train, x1_test, y1_train, y1_test = train_test_split(sentenceEmbed, y1, train_size=0.8, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(sentenceEmbed, y2, train_size=0.8, test_size=0.2)

#3.5

perceptron_classifier = MLPClassifier(max_iter=5)
print("We are using a Multi-Layered Perception Classifier or MLPClassifier with the default parameters.")
perceptron_classifier.fit(x1_train, y1_train)
y1_pred = perceptron_classifier.predict(x1_test)
print("Emotions: \n")
print(classification_report(y1_test, y1_pred))
print("Confusion Matrix: \n", confusion_matrix(y1_test, y1_pred))
perceptron_classifier = MLPClassifier(max_iter=20)
perceptron_classifier.fit(x2_train, y2_train)
y2_pred = perceptron_classifier.predict(x2_test)
print("Sentiments: \n")
print(classification_report(y2_test, y2_pred))
print("Confusion Matrix: \n", confusion_matrix(y2_test, y2_pred))

#3.6

# print("We are using a Multi-Layered Perception Classifier or MLPClassifier with the hyperparameters.")
# perceptron_classifier = MLPClassifier(max_iter=20, activation='relu', hidden_layer_sizes=(30,50), solver='sgd')
# print("We are using a Multi-Layered Perception Classifier or MLPClassifier with GridSearchCV.")
# perceptron_classifier.fit(x1_train,y1_train)
# y1_pred = perceptron_classifier.predict(x1_test)
# print("Emotions: \n")
# print(classification_report(y1_test,y1_pred))
# print("Confusion Matrix: \n")
#
# perceptron_classifier.fit(x2_train,y2_train)
# y2_pred = perceptron_classifier.predict(x2_test)
# print("Sentiments: \n")
# print(classification_report(y2_test, y2_pred))