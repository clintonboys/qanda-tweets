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

	right_count = 0
	left_count = 0

	keywords = []

	for right_term in right_list:
		if right_term in tweet_words:
			keywords.append(right_term)
			right_count += 1
	for left_term in left_list:
		if left_term in tweet_words:
			keywords.append(left_term)
			left_count += 1

	if (left_count >= 1 and right_count == 0) or (left_count == 0 and right_count >= 1):

		tweets_analysed +=1

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
			if left_count >= 1:
				print tweet, '| mentions left entity with keywords ' + ''.join(keywords) + ', '+str(np.round(prob_happy,3)) +'% of being positive'
				print '| sentiment derived from words ' + ''.join(sentiments)
				if prob_happy > 0.5:
					left_tweets +=1
				elif prob_happy <0.5:
					right_tweets +=1
			else:
				print tweet, '| mentions right entity with keywords ' +''.join(keywords) + ', ' +str(np.round(prob_happy,3)) +'% of being positive'
				print '| sentiment derived from words ' + ''.join(sentiments)
				if prob_happy >0.5:
					right_tweets +=1
				elif prob_happy <0.5:
					left_tweets +=1


print 'Total tweets: ' + str(tweet_total)
print 'Tweets analysed: ' + str(tweets_analysed) + ' (' +str(np.round(100*float(tweets_analysed)/float(tweet_total),1))+'%)'
print 'Left tweets: ' +str(left_tweets) + ' (' +str(np.round(100*float(left_tweets)/float(tweets_analysed),1))+'%)'
print 'Right tweets: ' +str(right_tweets) + ' (' +str(np.round(100*float(right_tweets)/float(tweets_analysed),1))+'%)'


