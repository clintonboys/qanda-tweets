## for lines that don't contain qanda (or QandA, etc etc)
## append them to previous lines 

## delete any line starting with RT

tweet_count = 0
bad_count = 0

with open('tweetdump27.txt') as f:
	tweets = f.readlines()

terms = ['#qanda', '#QandA', '@QandA', '@qanda', 'qanda', 'QandA', 'Qanda', '#Qanda', '@Qanda', '#QANDA']
termsn = [term + '\n' for term in terms]
termsp = [term + '.' for term in terms]
termsc = [term + ',' for term in terms]

search_terms = terms + termsn + termsp + termsc

print search_terms

def IsBad(tweet):
	if len([val for val in search_terms if val in tweet.split(' ')]) == 0:
		return True
	else:
		if tweet.split(' ')[0] == 'RT':
			return True
		else:
			return False

for tweet in tweets:
	tweet_count += 1
	if len([val for val in search_terms if val in tweet.split(' ')]) == 0:
		if len(tweet.split(' ')) % 18 == 0:
			print tweet.split(' ')
		bad_count +=1
	else:
		if tweet.split(' ')[0] == 'RT':
			bad_count += 1 

print tweet_count, bad_count

with open('tweetdump27_new.txt','wb') as f:
	for tweet in tweets:
		if not IsBad(tweet):
			f.write(tweet)
