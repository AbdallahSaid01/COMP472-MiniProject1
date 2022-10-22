from gensim.models import Word2Vec
import json as js
import numpy as np
import gensim.downloader as api
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction._dict_vectorizer import DictVectorizer


model = api.load('word2vec-google-news-300')

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