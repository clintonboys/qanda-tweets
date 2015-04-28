'''
TrainSentimemts.py
'''

import numpy as np

with open('trained_tweets.txt') as f:
	raw = f.readlines()

tweet_dict = {}

bad_count = 0
good_count = 0
neutral_count = 0
tweet_count = 0

for i in range(0,len(raw),2):
	tweet_count += 1
	if raw[i+1][0] == '0':
		tweet_dict[raw[i].replace('\n','').replace('.','').replace(',','').replace('?','').replace('!','').replace('#','').lower()] = -1
		bad_count += 1
	elif raw[i+1][0] == '1':
		tweet_dict[raw[i].replace('\n','').replace('.','').replace(',','').replace('?','').replace('!','').replace('#','').lower()] = 1
		good_count += 1
	else:
		tweet_dict[raw[i].replace('\n','').replace('.','').replace(',','').replace('?','').replace('!','').replace('#','').lower()] = 0
		neutral_count += 1

sentiment_dict = {}

for tweet in tweet_dict:
	for word in tweet.split(' '):
		try:
			sentiment_dict[word] = sentiment_dict[word] + tweet_dict[tweet]
		except KeyError:
			sentiment_dict[word] = tweet_dict[tweet]

with open('SentimentDict.txt', 'w') as f:
	for key in sorted(sentiment_dict):
		f.write(key)
		f.write(',')
		f.write(str(sentiment_dict[key]))
		f.write('\n')


print 'Tweets classified: ' + str(tweet_count) + ' (' +str(np.round(100*float(tweet_count)/float(11754+722),2))+'%)'
print 'Words classified: ' + str(len(sentiment_dict))
print 'Negative tweets: ' + str(np.round(100*float(bad_count)/float(tweet_count),2))
print 'Positive tweets: ' + str(np.round(100*float(good_count)/float(tweet_count),2))
print 'Neutral tweets and spam: ' + str(np.round(100*float(neutral_count)/float(tweet_count),2))