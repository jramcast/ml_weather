import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

csvfile = open('data/train.csv', newline='')

datareader = csv.DictReader(csvfile)


# Most simple params implementation, whether or not a certain work appers in the tweet
keywords = [
	"clouds",
	"cold",
	"dry",
	"hot",
	"humid",
	"hurricane",
	"tell",
	"ice",
	"other",
	"rain",
	"snow",
	"storms",
	"sun",
	"tornado",
	"wind",
	"now",
	"tomorrow",
	"yesterday",
	"current"
]


X = []
Y = []


print("Preparing data...")

for row in datareader:
	keywords_in_tweet = []
	state_in_tweet = 0
	location_in_tweet = 0

	# check whether each keyword is inside tweet
	for word in keywords:
		if word in row['tweet']:
			keywords_in_tweet.append(1)
		else:
			keywords_in_tweet.append(0)

	# check whether state is inside tweet
	if row['state'] in row['tweet']:
		state_in_tweet = 1
	
	# check whether location is inside tweet
	if row['location'] in row['tweet']:
		location_in_tweet = 1

	# now generate the numeric arrays x and y
	x_row = keywords_in_tweet + [state_in_tweet, location_in_tweet]
	y_row = [
		row['s1'], row['s2'], row['s3'], row['s4'], row['s5'], 
		row['w1'], row['w2'], row['w3'], row['w4'], 
		row['k1'], row['k2'], row['k3'], row['k4'], row['k5'], row['k6'], row['k7'], row['k8'],
		row['k9'], row['k10'], row['k11'], row['k12'], row['k13'], row['k14'], row['k15']          
	]

	y_row = [ round(float(val)) for val in y_row ]

	X.append(x_row)
	Y.append(y_row)


X = np.matrix(X)
Y = np.array(Y)

np.random.shuffle(X)
np.random.shuffle(Y)



m = X.shape[0]
validationset_start = round(m * 0.6)
testset_start = round(m * 0.8)


X_train = X[0:validationset_start, :]
Y_train = Y[0:validationset_start, :]
X_validation = X[validationset_start:testset_start, :]
Y_validation = Y[validationset_start:testset_start, :]
X_test = X[testset_start:, :]
Y_test = Y[testset_start:, :]


print("Training set size")
print(X_train.shape)
print("Validation set size")
print(X_validation.shape)
print("Test set size")
print(X_test.shape)

# Create a binary array marking values as True or False
from sklearn.preprocessing import MultiLabelBinarizer
Y_train = MultiLabelBinarizer().fit_transform(Y_train)



print("Training...") 	
print(Y_train)
lr = LogisticRegression(solver='newton-cg')
lr.fit(X_train, Y_train)

