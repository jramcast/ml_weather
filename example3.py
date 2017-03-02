"""
Test 3

Introduces the use of MLPClassifier.
"""
import csv
from random import shuffle
import numpy as np
from sklearn.neural_network import MLPClassifier
from frequent import get_most_frequent_terms


print("Reading CSV...")
csvfile = open('data/train.csv', newline='')
datareader = csv.DictReader(csvfile)
data = list(datareader)


print("Shuffling data...")
shuffle(data)
data = data[0: 10000]


HOW_MANY_FEATURES = 1000
print("Selecting features based on the most {} common words.".format(HOW_MANY_FEATURES))
most_frequent_terms = list()

def filter_tweets():
    only_training_tweets = data[0: int(len(data)*0.6)]
    for row in only_training_tweets:
        yield row['tweet']

most_frequent_terms = get_most_frequent_terms(filter_tweets(), HOW_MANY_FEATURES)
print(most_frequent_terms)
automatic_features = [text for text, times in most_frequent_terms]



print("Generating data features...")
X = []
y = []
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

    # for now, we only use the weather type class
    y_type_classes = [row['k1'], row['k2'], row['k3'], row['k4'], row['k5'], row['k6'], row['k7'], row['k8'],
        row['k9'], row['k10'], row['k11'], row['k12'], row['k13'], row['k14'], row['k15']]
    y_row = [ float(val) for val in y_type_classes ]

    # now generate the numeric arrays X
    x_row = keywords_in_tweet
    X.append(x_row)
    y.append(y_row)


print("Converting data to numpy matrix")
X = np.matrix(X)
y = np.matrix(y)

sigmoider = lambda val: 1 if float(val) >= 0.3 else 0
vsigmoid = np.vectorize(sigmoider)


print("Splitting data set in training, validation and test sets")
m = X.shape[0]
validationset_start = round(m * 0.6)
testset_start = round(m * 0.8)
X_train = X[0:validationset_start, :]
y_train = y[0:validationset_start]
X_validation = X[validationset_start:testset_start , :]
y_validation = y[validationset_start:testset_start]
X_test = X[testset_start:, :]


print("Training...")
classifier = MLPClassifier()
classifier.fit(X_train, vsigmoid(y_train))


def compute_error(X, Y, model, show_errors=False):
    """
    Computes the predicion error of a given model
    """
    error = 0
    m = X.shape[0]
    for i in range(0, m - 1):
        y_valid = Y[i]
        prediction = model.predict(X[i, :])
        #print('*******************')
        #print(y_valid)
        #print(prediction.astype(float))
        error = error + np.sum(np.square(prediction - y_valid))

    print('-- Total examples:', m)
    print('-- Cuadratic mean error:', (error / m))



print("Validating regularizacion term lambda...")
print('- Training set:')
compute_error(X_train, y_train, classifier)
print('- Validation set:')
compute_error(X_validation, y_validation, classifier)
