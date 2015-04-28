import pandas as pd
from pandas import DataFrame
import math
import csv
import numpy as np
import matplotlib.pyplot as plt

frame = pd.read_csv('SentimentDict.txt',skiprows=0,delim_whitespace=False,header=0,names=['term','rating'])
frame.index = frame['term']

with open('LeftList.txt', 'rb') as f:
	reader = csv.reader(f)
	left_list = list(reader)[0]

with open('RightList.txt', 'rb') as f:
	reader = csv.reader(f)
	right_list = list(reader)[0]

with open('tweetdump27_new.txt') as f:
	tweets = f.readlines()

print len(tweets)

def GetSentiment(tweet):
	score = 0
	for word in tweet.split(' '):
		try:
			score = score + frame['rating'][word]
		except KeyError:
			pass
	return score

scores_list = []
for tweet in tweets:
	if type(GetSentiment(tweet)) is np.int64:
		scores_list.append(float(GetSentiment(tweet))/float(len(tweet)))

plt.hist(scores_list,bins = 50)
plt.show()
# print tweets[9909]
# print np.round(float(GetSentiment(tweets[1101]))/float(len(tweets[9909].split(' '))),2)

# print np.mean(scores_list)
# print max(scores_list)
# print min(scores_list)
