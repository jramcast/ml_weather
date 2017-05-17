"""
Test 8
"""
import csv
import time
import math
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
from preprocessing import (SentimentExtractor,
                           TempExtractor,
                           WindExtractor,
                           tokenize,
                           stopwords_list)

print("Reading CSV...")
csvfile = open('data/train.csv', newline='')
datareader = csv.DictReader(csvfile)
data = list(datareader)

classes = [
    'k1',
    'k2',
    'k3',
    'k4',
    'k5',
    'k6',
    'k7',
    'k8',
    'k9',
    'k10',
    'k11',
    'k12',
    'k13',
    'k14',
    'k15',
    #'s1',
    's2',
    's3',
    's4',
    's5',
    'w1',
    'w2',
    #'w3',
    'w4',
]

def filter_tweets(data):
    return [row['tweet'] for row in data]

def filter_class(data, className):
    y = []
    for row in data:
        value = float(row[className])
        y.append(math.ceil(value))

    return y

x_train = filter_tweets(data)

sentiment_extractor = SentimentExtractor()
temp_extractor = TempExtractor()
wind_extractor = WindExtractor()

for className in classes:
    y_train = filter_class(data, className)
    print("--------------------------> Training " + className)
    start_time = time.time()

    vectorizer = CountVectorizer(
        min_df=10,
        max_df=0.5,
        ngram_range=(1, 3),
        max_features=10000,
        lowercase=True,
        stop_words=stopwords_list,
        tokenizer=tokenize
    )

    svm = LinearSVC(C=0.3, max_iter=300, loss='hinge')
    pipeline = Pipeline([
        ('union', FeatureUnion([
            ('sentiment', sentiment_extractor),
            ('temp', temp_extractor),
            ('wind', wind_extractor),
            ('vect', vectorizer),
        ])),
        ('cls', svm),
    ])

    y_train = filter_class(data, className)
    accuracy = cross_val_score(pipeline, x_train, y_train, scoring='accuracy')
    print('=== Accuracy ===')
    print(np.mean(accuracy))
    print('Time elapsed:')
    pipeline.fit(x_train, y_train)
    print(time.time() - start_time)
    # save model
    joblib.dump(pipeline, 'models/{}.pkl'.format(className))
