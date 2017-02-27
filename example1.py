"""
Test 1

Calculate Error on training and validation sets manually.

Quick and dirty test to classify tweets talking about weather.
"""

import csv
import numpy as np
from random import shuffle
from sklearn.linear_model import LogisticRegression
from frequent import get_most_frequent_terms

# Prepare future numpy matrices
X = []
Y_sentiment = []
Y_when = []
Y_type = []
#lambda regularization param
lambda_reg = 1
# solver for minification optimization
solver = 'liblinear'
# Structures to hold original data
# We want to store original data to perform manual error analysis on some tweets
original_data = []
original_data_key = 0


print("Reading CSV...")
csvfile = open('data/train.csv', newline='')
datareader = csv.DictReader(csvfile)
data = list(datareader)


print("Shuffling data...")
shuffle(data)
#data = data[0: 10000]


how_many_features = 500
print("Getting most {} common words as features.".format(how_many_features))
most_frequent_terms = list()
def filter_tweets():
    only_training_tweets = data[0: int(len(data)*0.6)]
    for row in only_training_tweets:
        yield row['tweet']
most_frequent_terms = get_most_frequent_terms(filter_tweets(), how_many_features)
print(most_frequent_terms)
automatic_features = [text for text, times in most_frequent_terms]

print("Generating data features...")
for row in data:
    keywords_in_tweet = []
    state_in_tweet = 0
    location_in_tweet = 0

    # check whether each keyword is inside tweet
    tweet = row['tweet'].lower()
    for keyword in automatic_features:
        if keyword in tweet:
            keywords_in_tweet.append(1)
        else:
            keywords_in_tweet.append(0)

    # check whether state is inside tweet
    if row['state'] in row['tweet']:
        state_in_tweet = 1

    # check whether location is inside tweet
    if row['location'] in row['tweet']:
        location_in_tweet = 1

    # each row must have 3 classes: sentiment, when and type
    # we found the class of each row by looking a the one with max value
    y_sentiment_classes = [row['s1'], row['s2'], row['s3'], row['s4'], row['s5']]
    y_when_classes = [row['w1'], row['w2'], row['w3'], row['w4']]
    y_type_classes = [row['k1'], row['k2'], row['k3'], row['k4'], row['k5'], row['k6'], row['k7'], row['k8'],
        row['k9'], row['k10'], row['k11'], row['k12'], row['k13'], row['k14'], row['k15']]
    # we sum 1 to have 1-indexed classes, e.g 1 equals s1
    y_sentiment = y_sentiment_classes.index(max(y_sentiment_classes)) + 1
    y_when = y_when_classes.index(max(y_when_classes)) + 1
    y_type = y_type_classes.index(max(y_type_classes)) + 1

    # now generate the numeric arrays x and y
    x_row = [original_data_key] + keywords_in_tweet + [state_in_tweet, location_in_tweet, y_sentiment, y_when, y_type]
    X.append(x_row)

    # Store the original example in a dictionary for future exploration
    row['classes'] = ("s{}".format(y_sentiment), "w{}".format(y_when), "k{}".format(y_type))
    row['data_key'] = original_data_key
    original_data.append(row)
    original_data_key = original_data_key + 1


print("Converting data to numpy matrix")
X = np.matrix(X)


print("Splitting data set in training, validation and test sets")
m = X.shape[0]
# TODO: change test validation set start to 60% when we start using x_test
validationset_start = round(m * 0.6)
testset_start = round(m * 0.8)
X_train = X[0:validationset_start, :]
X_validation = X[validationset_start:testset_start , :]
X_test = X[testset_start:, :]


print("Separating features(X) and classes(Y)")


def separate_X_and_Y(X):
    m = X.shape[0]
    n = X.shape[1]
    Y_sentiment = np.ravel(X[:, n-3])
    Y_when = np.ravel(X[:, n-2])
    Y_type = np.ravel(X[:, n-1])
    X = X[:, 0:n-3]
    return X, Y_sentiment, Y_when, Y_type


X_train, Y_sentiment_train, Y_when_train, Y_type_train = separate_X_and_Y(X_train)
X_validation, Y_sentiment_validation, Y_when_validation, Y_type_validation = separate_X_and_Y(X_validation)



def compute_error(X, Y, model, show_errors=False):
    """
    Computes the predicion error of a given model
    """
    error = 0;
    m = X.shape[0]
    for i in range(0, m - 1):
        # Drop first column (example id) as it is useless for predicting
        prediction = model.predict(X[i, 1:]).item(0)
        y_valid = Y[i]
        if prediction != y_valid:
            error = error + 1
            example_id = X[i, 0]
            if show_errors:
                print(original_data[example_id])
                print("Valid class: k{}".format(y_valid))
                print("Predicted class: k{}".format(prediction))
    print('-- Total examples:', m)
    print('-- Total errors:', error)
    print('-- Cuadratic mean error:', (error / m))



print("Validating regularizacion term lambda...")
for lambda_param in [ 0.01, 0.03, 0.09, 0.1, 0.3, 0.9, 1, 3, 9 ]:
    print(" ")
    print("-------- Training with lambda: ", lambda_param)
    model = LogisticRegression(C=1/lambda_param, solver=solver)
    # Drop first column (example id) as it is useless for predicting
    model.fit(X_train[:, 1:], np.ravel(Y_type_train))
    print('- Training set:')
    compute_error(X_train, Y_type_train, model)
    print('- Validation set:')
    compute_error(X_validation, Y_type_validation, model)

