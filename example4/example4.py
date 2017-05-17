"""
Test 4

Introduces the use of MLPClassifier, with multilabels.
"""
import csv
import math
from random import shuffle
import numpy as np
import string
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from frequent import get_most_frequent_terms
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


# Prepare stopwords
punctuation = list(string.punctuation)
stopwords_list = stopwords.words('english') + punctuation \
                    + ['rt', 'via', '…', '...', '️', 'ヽ', '、', '｀' ]


print("Reading CSV...")
csvfile = open('data/train.csv', newline='')
datareader = csv.DictReader(csvfile)
data = list(datareader)


print("Shuffling data...")
shuffle(data)
data = data[0: 10000]


HOW_MANY_FEATURES = 1000
print("Selecting features based on the most {} common words.".format(HOW_MANY_FEATURES))
tokenizer = TweetTokenizer(preserve_case=True)
vectorizer = TfidfVectorizer(
    min_df=1, 
    max_features=HOW_MANY_FEATURES,
    ngram_range=(1, 3),
    stop_words=stopwords_list,
    #tokenizer=tokenizer.tokenize
)

def filter_tweets(data):
    for row in data:
        yield row['tweet']

print("Generating data features...")
X = vectorizer.fit_transform(filter_tweets(data))
print(vectorizer.get_feature_names())
X = X.toarray()
y = []
for row in data:
    # for now, we only use the weather type class
    y_type_classes = [row['k1'], row['k2'], row['k3'], row['k4'], row['k5'], row['k6'], row['k7'], row['k8'],
        row['k9'], row['k10'], row['k11'], row['k12'], row['k13'], row['k14'], row['k15']]
    y_row = [ float(val) for val in y_type_classes ]
    y.append(y_row)


print("Converting data to numpy matrix")
X = np.matrix(X)
y = np.matrix(y)

sigmoider = lambda val: 1 if float(val) >= 0.3 else 0

vsigmoid = np.vectorize(sigmoider)

print("Training...")
classifier = MLPClassifier(hidden_layer_sizes=(2000, 1000, 500))
precision = cross_val_score(classifier, X, vsigmoid(y), scoring='precision_weighted')
recall = cross_val_score(classifier, X, vsigmoid(y), scoring='recall_weighted')

meanprecision = np.mean(precision)
meanrecall = np.mean(recall)
print('Precision')
print(meanprecision)
print('Recall')
print(meanrecall)
print('F1')

print(2 * (meanprecision * meanrecall) / (meanprecision + meanrecall) )

