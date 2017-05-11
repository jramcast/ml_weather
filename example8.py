"""
Test 4

Introduces the use of MLPClassifier, with multilabels.
"""
import csv
import time
import string
import re
import math
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, Imputer, label_binarize
from sklearn.feature_extraction import DictVectorizer
from textblob import TextBlob
from sklearn.externals import joblib

print("Reading CSV...")
csvfile = open('data/train.csv', newline='')
datareader = csv.DictReader(csvfile)
data = list(datareader)


class NumericFieldsExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts all numeric features from tweets
    """

    def __init__(self):
        pass

    def transform(self, tweets, y_train=None):
        samples = []
        for tweet in tweets:
            textBlob = TextBlob(tweet)
            samples.append({
                'temp': self.get_temperature(tweet),
                'wind': self.get_wind(tweet),
                'sent_polarity': textBlob.sentiment.polarity,
                'sent_subjetivity': textBlob.sentiment.subjectivity
            })
        vectorized = DictVectorizer().fit_transform(samples).toarray()
        vectorized = Imputer().fit_transform(vectorized)
        vectorized_scaled = MinMaxScaler().fit_transform(vectorized)
        return vectorized_scaled

    def fit(self, X, y=None):
        return self

    def get_temperature(self, tweet):
        match = re.search(r'(\d+(\.\d)?)\s*F', tweet, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            celsius = (value - 32) / 1.8
            if - 100 < celsius < 100:
                return celsius
        return None

    def get_wind(self, tweet):
        match = re.search(r'(\d+(\.\d)?)\s*mph', tweet, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            kph = value * 1.60934
            if 0 <= kph < 500:
                return kph
        return None


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

    sentence = TextBlob(text)
    tokens = [word.lemmatize() for word in sentence.words]
    return tokens

    """tokens = tokenizer.tokenize(text)
    # stem
    try:
        stems = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems"""


def filter_tweets(data):
    return [ row['tweet'] for row in data ]



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
def filter_class(data, className):
    y = []
    for row in data:
        value = float(row[className])
        y.append(math.ceil(value))

    return y

x_train = filter_tweets(data)

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
    num_extractor = NumericFieldsExtractor()
    svm = LinearSVC(C=0.3, max_iter=300, loss='hinge')
    pipeline = Pipeline([
        ('union', FeatureUnion([
            ('vect', vectorizer),
            ('nums', num_extractor),
        ])),
        ('cls', svm),
    ])

    y_train = filter_class(data, className)
    accuracy = cross_val_score(pipeline, x_train, y_train, scoring='accuracy')
    print('=== Accuracy ===')
    print(np.mean(accuracy))
    print('Time elapsed:')
    print(time.time() - start_time)
    # save model
    joblib.dump(pipeline, 'models/{}.pkl'.format(className))
