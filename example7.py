# Temp
#(\d+(\.\d)?)\s*F

# Speed
#(\d+(\.\d)?)\s*mph


"""
Test 4

Introduces the use of MLPClassifier, with multilabels.
"""
import csv
import math
import time
import numpy as np
import string
import matplotlib
import re
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
csvfile = open('data/test.csv', newline='')
datareader = csv.DictReader(csvfile)
data = list(datareader)


regex_temp = r'(\d+(\.\d)?)\s*F'

def filter_tweets(data):
    count = 0
    for row in data:
        tweet = row['tweet']
        m = re.search(regex_temp, tweet, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            celsius = (value - 32) / 1.8
            if (- 100 < celsius < 100):
                print(celsius)
                count = count + 1

    print ('count')
    print (count)
    print ('total')
    print (len(data))




x_train = filter_tweets(data)

