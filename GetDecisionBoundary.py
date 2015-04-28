import pandas as pd
from pandas import DataFrame
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LinearRegression

with open('trained_tweets.txt') as f:
	raw = f.readlines()

frame = pd.read_csv('SentimentDict.txt',skiprows=0,delim_whitespace=False,header=0,names=['term','rating'])
frame.index = frame['term']

tweet_dict = {}

bad_count = 0
good_count = 0
neutral_count = 0
tweet_count = 0
tweets_final = []

for i in range(0,len(raw),2):
	tweet_count += 1
	if raw[i+1][0] == '0':
		tweet_dict[raw[i].replace('\n','').replace('.','').replace(',','').replace('?','').replace('!','').replace('#','').lower()] = -1
		bad_count += 1
	elif raw[i+1][0] == '1':
		tweet_dict[raw[i].replace('\n','').replace('.','').replace(',','').replace('?','').replace('!','').replace('#','').lower()] = 1
		good_count += 1
	else:
		pass

def GetSentiment(tweet):
	score = 0
	for word in tweet.split(' '):
		try:
			score = score + frame['rating'][word]
		except KeyError:
			pass
	return score

def GetSentiment2(tweet):
	total_words = 0
	bad_words = 0
	for word in tweet.split(' '):
		try:
			if frame['rating'][word] < 0:
				total_words += 1
				bad_words +=1
			else:
				total_words += 1
		except KeyError:
			pass
	return float(bad_words) / float(total_words)

sent_dict = {}
scores_list = []
scores_list2 = []
classes_list = []

good_scores = []
bad_scores = []

for tweet in tweet_dict:
	if type(GetSentiment(tweet)) is np.int64:
		sent_dict[tweet] = float(GetSentiment(tweet))/float(len(tweet))
		scores_list.append(float(GetSentiment(tweet))/float(len(tweet)))
		scores_list2.append(float(GetSentiment2(tweet)))
		classes_list.append(float(tweet_dict[tweet]))
		tweets_final.append(tweet.replace('\n','').replace('.','').replace(',','').replace('?','').replace('!','').replace('#','').lower())
		if tweet_dict[tweet] == 1:
			good_scores.append(float(GetSentiment(tweet))/float(len(tweet)))
		elif tweet_dict[tweet] == -1:
			bad_scores.append(float(GetSentiment(tweet))/float(len(tweet)))

def AccuracyScore(boundary): #boundary1 < boundary2
	correct_count = 0
	for tweet in tweet_dict:
		try:
			if (sent_dict[tweet] <= boundary) and (tweet_dict[tweet] == -1):
				correct_count += 1
			elif (boundary < sent_dict[tweet]) and (tweet_dict[tweet] == 1):
				correct_count += 1
		except KeyError:
			pass
	return float(correct_count) / float(tweet_count)

data = pd.DataFrame([tweets_final, scores_list, scores_list2,classes_list]).transpose()
data.columns = ['tweet', 'normsent', 'negperc', 'class']

print data.head()

X = data[['normsent', 'negperc']].values
y = data['class']

est = LinearRegression(fit_intercept = True, normalize = True)
est.fit(X,y)


color_map = []
for score in classes_list:
	if score == 1:
		color_map.append('b')
	else:
		color_map.append('r')


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


ax = plt.gca()
ax.scatter(data.normsent, data.negperc, c =color_map)
plot_surface(est, X[:,0], X[:,1],ax =ax)
plt.show()

# h=0.01

# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))

clf = svm.SVC()
print clf.fit([[scores_list[i],scores_list2[i]] for i in range(0,len(scores_list))],classes_list)

print np.mean(good_scores)
print np.mean(bad_scores)
#plt.contourf(xx,yy,Z)
# plt.scatter(scores_list,scores_list2,color=color_map)
# plt.show()

# plt.hist(scores_list,bins = 50)
# plt.show()

# def GetBoundaries():

# 	# start with decent guess (gives .413)

# 	boundary1 = -2
# 	boundary2 = -1
# 	accuracy = 0.413

# 	for test_boundary1 in np.arange(-5,0,0.01):
# 		for test_boundary2 in np.arange(test_boundary1,5,0.01):
# 			if AccuracyScore(test_boundary1,test_boundary2) > accuracy:
# 				boundary1 = test_boundary1
# 				boundary2 = test_boundary2
# 				accuracy = AccuracyScore(test_boundary1,test_boundary2)

# 	return boundary1, boundary2, accuracy






