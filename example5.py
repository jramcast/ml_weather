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
from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from frequent import get_most_frequent_terms
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Prepare stopwords
punctuation = list(string.punctuation)
stopwords_list = stopwords.words('english') + punctuation \
                    + ['rt', 'via', '…', '...', '️', 'ヽ', '、', '｀', '_' ]


print("Reading CSV...")
csvfile = open('data/train.csv', newline='')
datareader = csv.DictReader(csvfile)
data = list(datareader)



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

print("Setting threshold for positive classes")
sigmoider = lambda val: 1 if float(val) >= 0.5 else 0
vsigmoid = np.vectorize(sigmoider)

print("Training...")
start_time = time.time()
"""
param_range = [0.03, 0.1, 0.3, 1, 3]
classifier = OneVsRestClassifier(LinearSVC())
train_scores, test_scores = validation_curve(
    classifier, X, vsigmoid(y), 
    param_name="estimator__C", param_range=param_range, scoring="accuracy", n_jobs=4
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel("C")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
print(time.time() - start_time)




plt.show()
"""

classifier = OneVsRestClassifier(LinearSVC(C=0.3))
accuracy = cross_val_score(classifier, X, vsigmoid(y), scoring='accuracy')
precision = cross_val_score(classifier, X, vsigmoid(y), scoring='precision_weighted')
recall = cross_val_score(classifier, X, vsigmoid(y), scoring='recall_weighted')

print('Accuracy')
print(accuracy)
meanprecision = np.mean(precision)
meanrecall = np.mean(recall)
print('Precision')
print(meanprecision)
print('Recall')
print(meanrecall)
print('F1')
print(2 * (meanprecision * meanrecall) / (meanprecision + meanrecall) )
print ('Time elapsed:')
print(time.time() - start_time)

