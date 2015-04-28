import pandas as pd
from pandas import DataFrame
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LinearRegression

## Open the training data and test data

with open('trained_tweets.txt') as f:
	trained_tweets = f.readlines()

with open('tweetdump27_new.txt') as f:
	test_tweets = f.readlines()

## Open the sentiment dictionary

sentiment_dict = pd.read_csv('SentimentDict.txt',skiprows=0,delim_whitespace=False,header=0,names=['term','rating'])
sentiment_dict.index = sentiment_dict['term']

## Clean up the classifications on the trained data, ignoring 
## tweets classified as neutral or spam. 

tweet_dict = {}

bad_count = 0
good_count = 0
tweet_count = 0

for i in range(0,len(trained_tweets),2):
	tweet_count += 1
	if trained_tweets[i+1][0] == '0':
		tweet_dict[trained_tweets[i].replace('\n','').replace('.','').replace(',','').replace('?','').replace('!','').replace('#','').lower()] = -1
		bad_count += 1
	elif trained_tweets[i+1][0] == '1':
		tweet_dict[trained_tweets[i].replace('\n','').replace('.','').replace(',','').replace('?','').replace('!','').replace('#','').lower()] = 1
		good_count += 1
	else:
		pass

def NormalisedSentiment(tweet):
	score = 0
	for word in tweet.split(' '):
		try:
			score = score + sentiment_dict['rating'][word]
		except KeyError:
			pass
	return score

def NegativePercentage(tweet):
	total_words = 0
	bad_words = 0
	for word in tweet.split(' '):
		try:
			if sentiment_dict['rating'][word] < 0:
				total_words += 1
				bad_words +=1
			else:
				total_words += 1
		except KeyError:
			pass
	return float(bad_words) / float(total_words)

## Calculate the two variables for the training data 
## and put everything into a neat dataframe. 

tweets_final = []
normsent = []
negperc = []
classes = []

for tweet in tweet_dict:
	if type(NormalisedSentiment(tweet)) is np.int64:
		tweets_final.append(tweet)
		normsent.append(float(NormalisedSentiment(tweet))/float(len(tweet)))
		negperc.append(NegativePercentage(tweet))
		classes.append(float(tweet_dict[tweet]))

training_data = pd.DataFrame([tweets_final, normsent, negperc, classes]).transpose()
training_data.columns = ['tweet', 'normsent', 'negperc', 'class']

print training_data.head()

## Fit a linear regression model to the training data

X = training_data[['normsent', 'negperc']].values
y = training_data['class']

print y

est = svm.SVC(kernel = 'linear')
linear_model = est.fit(X,y)

color_map = []
for score in classes:
	if score == 1:
		color_map.append('b')
	else:
		color_map.append('r')

## Code to plot decision boundary

def plot_surface(est, x_1, x_2, ax=None, threshold=0.0, contourf=False):
    """Plots the decision surface of ``est`` on features ``x1`` and ``x2``. """
    xx1, xx2 = np.meshgrid(np.linspace(x_1.min(), x_1.max(), 100), 
                           np.linspace(x_2.min(), x_2.max(), 100))
    # plot the hyperplane by evaluating the parameters on the grid
    X_pred = np.c_[xx1.ravel(), xx2.ravel()]  # convert 2d grid into seq of points
    if hasattr(est, 'predict_proba'):  # check if ``est`` supports probabilities
        # take probability of positive class
        pred = est.predict_proba(X_pred)[:, 1]
    else:
        pred = est.predict(X_pred)
    Z = pred.reshape((100, 100))  # reshape seq to grid
    if ax is None:
        ax = plt.gca()
    # plot line via contour plot
    
    if contourf:
        ax.contourf(xx1, xx2, Z, levels=np.linspace(0, 1.0, 10), cmap=plt.cm.RdBu, alpha=0.6)
    ax.contour(xx1, xx2, Z, levels=[threshold], colors='black')
    ax.set_xlim((x_1.min(), x_1.max()))
    ax.set_ylim((x_2.min(), x_2.max()))

# ax = plt.gca()
# ax.scatter(training_data.normsent, training_data.negperc, c =color_map)
# plot_surface(est, X[:,0], X[:,1],ax =ax)
# plt.show()

## Load test data

tweets_final_test = []
normsent_test = []
negperc_test = []

for tweet in test_tweets:
	if type(NormalisedSentiment(tweet)) is np.int64:
		tweets_final_test.append(tweet)
		normsent_test.append(float(NormalisedSentiment(tweet))/float(len(tweet)))
		negperc_test.append(NegativePercentage(tweet))

test_data = pd.DataFrame([tweets_final_test, normsent_test, negperc_test]).transpose()
test_data.columns = ['tweet', 'normsent', 'negperc']

print test_data.head()

Xt = test_data[['normsent', 'negperc']].values

predicted_classes = linear_model.predict(Xt)

pred_color_map = []
for score in predicted_classes:
	if score == -1.0:
		pred_color_map.append('b')
	else:
		pred_color_map.append('r')


test_data['classes'] = predicted_classes

for i in range(0,50):
	print test_data['tweet'][i], test_data['classes'][i]


with open('LeftList.txt', 'rb') as f:
	reader = csv.reader(f)
	left_list = list(reader)[0]

with open('RightList.txt', 'rb') as f:
	reader = csv.reader(f)
	right_list = list(reader)[0]

alignment = []

for i in range(0,len(test_data)):
	left_words = 0
	right_words = 0
	for word in test_data['tweet'][i].split(' '):
		if word in left_list:
			left_words += 1
		elif word in right_list:
			right_words += 1
	if left_words > right_words:
		alignment.append(1)
	elif right_words > left_words:
		alignment.append(-1)
	else:
		alignment.append(0)

test_data['alignment'] = alignment

left_count = 0
right_count = 0
tweet_count = 0

full_color_map = []
for i in range(0,len(test_data)):
	if test_data['classes'][i] == -1.0:
		if test_data['alignment'][i] == 1:
			full_color_map.append('b')
			right_count += 1
			tweet_count += 1
		elif test_data['alignment'][i] == -1:
			full_color_map.append('r')
			left_count += 1
			tweet_count += 1
		else:
			full_color_map.append('w')
			tweet_count +=1 
	else:
		if test_data['alignment'][i] == 1:
			full_color_map.append('r')
			left_count += 1
			tweet_count += 1
		elif test_data['alignment'][i] == -1:
			full_color_map.append('b')
			right_count += 1
			tweet_count += 1
		else:
			full_color_map.append('w')
			tweet_count += 1

plt.scatter(test_data.normsent, test_data.negperc, c =full_color_map)
plt.show()

print 'Left tweets: ' + str(np.round(100*float(left_count)/float(tweet_count),2))+'%'
print 'Right tweets: ' + str(np.round(100*float(right_count)/float(tweet_count),2)) + '%'
print 'Total tweets: ' + str(tweet_count)