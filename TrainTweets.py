'''
Want to be able to execute
python TrainTweets.py
and to start classifying tweets as 
0 - Negative
1 - Positive
9 - Spam
and to write the results to a .csv file. 
I also want to be able to pick up where I left off.

Ideally:

when the code is run, it opens a file containing the 
remaining tweet data that hasn't been classified yet. 
It then outputs csv'd tweets and user-specified 
ratings to the end of the final file. It then deletes
these lines from the top of the original file. 
'''

import numpy as np

with open('test_tweets2.txt') as f:
	tweets = f.readlines()
classified_dict = {}
for tweet in tweets:
	classified_dict[tweet] = False
tweets_total = len(classified_dict)


def main():

	## Keep two files open: test_tweets.txt which contains the unclassified 
	## tweet data, and trained_tweets.txt which contains the classified
	## tweet data. 

	with open('trained_tweets.txt', 'a') as g:
		count = 0
		for tweet in tweets:
			print tweet
			classification = raw_input('Classify this tweet as negative (0), positive (1), neutral(2) or spam (9): ')
			g.write(tweet.replace(',','') + str(classification))
			g.write('\n')
			classified_dict[tweet] = True
			print '-----------------------------------------------'
			count = count + 1
			if count%40 == 0:
				print str(np.round(100*float(count)/float(tweets_total),2))+'% completed...'
				print '-----------------------------------------------'

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        '''Do clever things here to make sure the
           files are preserved and not deleted. '''
        with open('test_tweets.txt', 'w') as f:
        	for tweet in classified_dict:
        		if classified_dict[tweet] == False:
        			f.write(tweet)