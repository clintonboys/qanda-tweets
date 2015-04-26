import pandas as pd
from pandas import DataFrame
import math
import csv
import numpy as np

frame = pd.read_csv('twitter_sentiment_list.csv',skiprows=0,delim_whitespace=False,header=0,names=['term','pos','neg'])
frame.index = frame['term']

with open('left_list.txt', 'rb') as f:
	reader = csv.reader(f)
	left_list = list(reader)[0]

with open('right_list.txt', 'rb') as f:
	reader = csv.reader(f)
	right_list = list(reader)[0]

print right_list

#tweets = ['I dont understand how the Coalition can be so cruel towards asylum seekers #QandA',
#			'PM very smart and good and sensible @QandA #QandA',
#			'Burke my favorite happy wonderful love politician in the world',
#			'awful bad horrible labor party: hasnt done anything, never again! #QandA']

with open('tweetdump.txt') as f:
	tweets = f.readlines()

tweet_total = 0
tweets_analysed = 0
left_tweets = 0
right_tweets = 0

for tweet in tweets:

	tweet_total += 1

	tweet_words = tweet.split(' ')

	happy_tweets = 0
	sad_tweets = 0

	keywords = []

	pos_sentiment = 0
	neg_sentiment = 0

	sentiments = []

	for word in tweet_words:
		if word in frame['term'].values:
			sentiments.append(word)
			pos_sentiment = pos_sentiment + frame['pos']['term']
			neg_sentiment = neg_sentiment + frame['neg']['term']
	prob_happy = np.reciprocal(np.exp(neg_sentiment - pos_sentiment)+1)
	prob_sad = 1-prob_happy
	if len(sentiments) > 0:
		tweets_analysed +=1
	if prob_sad > 0.5:
		sad_tweets += 1
	elif prob_happy > 0.5:
		happy_tweets += 1


print 'Total tweets: ' + str(tweet_total)
print 'Tweets analysed: ' + str(tweets_analysed) + ' (' +str(np.round(100*float(tweets_analysed)/float(tweet_total),1))+'%)'
print 'Positive tweets: ' + str(happy_tweets) + ' (' +str(np.round(100*float(happy_tweets)/float(tweets_analysed),1)) + '%)'

