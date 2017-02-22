import csv
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# The tokeninzer converts to lowercase and
# reduces lenght. For example: waaaayyyy to way
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)


# Most simple params implementation, whether or not a certain work appers in the tweet
from settings import keywords
stemmer = PorterStemmer()
stemmed_keywords = [ stemmer.stem(keyword) for keyword in keywords]


X = []
Y_sentiment = []
Y_when = []
Y_type = []


print("Preparing non numerical data...")
csvfile = open('data/train.csv', newline='')
datareader = csv.DictReader(csvfile)

original_data = []
original_data_key = 0

for row in datareader:
    keywords_in_tweet = []
    state_in_tweet = 0
    location_in_tweet = 0

    # check whether each keyword is inside tweet
    tweet = row['tweet'].lower()
    tweet_tokens = [ stemmer.stem(word) for word in tokenizer.tokenize(tweet)]

    for keyword in stemmed_keywords:
        if keyword in tweet_tokens:
            keywords_in_tweet.append(1)
        else:
            keywords_in_tweet.append(0)

    for keyword in keywords:
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
    row['stemmed'] = tweet_tokens
    row['data_key'] = original_data_key
    original_data.append(row)
    original_data_key = original_data_key + 1


print("Converting data to numpy matrix")
X = np.matrix(X)
# Limit examples until the model is more tuned
X = X[0:2000, :]

print("Shuffling data...")
np.random.shuffle(X)


print("Splitting data set in training, validation and test sets")
m = X.shape[0]
validationset_start = round(m * 0.6)
testset_start = round(m * 0.8)
X_train = X[0:validationset_start, :]
X_validation = X[validationset_start:testset_start, :]
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


print("Training...")
lr_type = LogisticRegression()
# Drop first column (example id) as it is useless for predicting
lr_type.fit(X_train[:, 1:], np.ravel(Y_type_train))


print("Validating...")

def compute_error(X, Y, model, show_errors=False):
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
                print(' ')
                print(original_data[example_id])
                print("Valid class: k{}".format(y_valid))
                print("Predicted class: k{}".format(prediction))
    print(' ')
    print('Total examples:', m)
    print('Total errors:', error)
    print('Cuadratic mean error:', (error / m))


print('- TRAINING SET ERROR')
compute_error(X_train, Y_type_train, lr_type, show_errors=True)

print('')
print('- VALIDATION SET ERROR')
compute_error(X_validation, Y_type_validation, lr_type)


"""
This test bellow is a more sophisticaed way to validate the model
and plot learning curves 
"""
print('')
print('- LEARNING CURVES')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Cross validation with 3 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)

n = X.shape[1]
X_curve = X[:, 0:n-3]
y_curve = np.ravel(X[:, n-1])

estimator = LogisticRegression()
plot_learning_curve(estimator, 'Learning curve', X_curve, y_curve, ylim=(0.3, 1.01), cv=cv, n_jobs=1)
plt.show()
