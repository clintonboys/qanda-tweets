'''
TrainSentimemts.py
'''

with open('trained_tweets.txt') as f:
	raw = f.readlines()

tweet_dict = {}

for i in range(0,len(raw),2):
	if raw[i+1][0] == '0':
		tweet_dict[raw[i]] = -1
	elif raw[i+1][0] == '1':
		tweet_dict[raw[i]] = 1
	else:
		tweet_dict[raw[i]] = 0

sentiment_dict = {}

for tweet in tweet_dict:
	for word in tweet.split(' '):
		try:
			sentiment_dict[word] = sentiment_dict[word] + tweet_dict[tweet]
		except KeyError:
			sentiment_dict[word] = tweet_dict[tweet]