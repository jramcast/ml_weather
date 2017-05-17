"""
Test 6

Introduces the use of MLPClassifier, with multilabels.
"""
import csv
import math
import time
import numpy as np
import string
import matplotlib
from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from frequent import get_most_frequent_terms
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline

print("Reading CSV...")
csvfile = open('data/train.csv', newline='')
datareader = csv.DictReader(csvfile)
data = list(datareader)

stopwords_list = stopwords.words('english')

non_words = list(string.punctuation)
non_words.extend(['¿', '¡'])

stemmer = SnowballStemmer('english')
tokenizer = TweetTokenizer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    text = ''.join([c for c in text if c not in non_words])
    tokens = tokenizer.tokenize(text)
    # stem
    try:
        stems = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems


HOW_MANY_FEATURES = 5
print("Selecting features based on the most {} common words.".format(HOW_MANY_FEATURES))
vectorizer = CountVectorizer(
    lowercase=True,
    stop_words=stopwords_list,
    tokenizer=tokenize
)

pipeline = Pipeline([
    ('vect', vectorizer),
    ('cls', OneVsRestClassifier(LinearSVC())),
])


def filter_tweets(data):
    return [ row['tweet'] for row in data ]

def filter_classes(data):
    y = []
    for row in data:
        # for now, we only use the weather type class
        y_type_classes = [row['k1'], row['k2'], row['k3'], row['k4'], row['k5'], row['k6'], row['k7'], row['k8'],
            row['k9'], row['k10'], row['k11'], row['k12'], row['k13'], row['k14'], row['k15']]
        y_row = [ float(val) for val in y_type_classes ]
        y_row = y_row.index(max(y_row)) + 1
        y.append(y_row)
    classes = [
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    ]
    return label_binarize(y, classes)

x_train = filter_tweets(data)
y_train = filter_classes(data)

parameters = {
    'vect__max_df': (0.5, 1.9),
    'vect__min_df': (10, 20, 50),
    'vect__max_features': (1000, 3000, 10000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigramas or bigramas
    'cls__estimator__C': (0.3, 1, 3),
    'cls__estimator__loss': ('hinge', 'squared_hinge'),
    'cls__estimator__max_iter': (300, 1000, 3000)
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=6, scoring='accuracy')
grid_search.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(grid_search.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))


print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
#y_true, y_pred = y_test, clf.predict(X_test)
#print(classification_report(y_true, y_pred))
print()

# Best: 0.786108 using {'vect__min_df': 10, 'cls__estimator__C': 0.3, 'vect__max_features': 10000, 'cls__estimator__max_iter': 300, 'vect__max_df': 0.5, 'cls__estimator__loss': 'hinge', 'vect__ngram_range': (1, 3)}