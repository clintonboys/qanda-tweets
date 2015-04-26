import tweepy
from tweepy.streaming import StreamListener
from tweepy import Stream
from keys import keys

CONSUMER_KEY = keys['consumer_key']
CONSUMER_SECRET = keys['consumer_secret']
ACCESS_TOKEN = keys['access_token']
ACCESS_TOKEN_SECRET = keys['access_token_secret']

# auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
# auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
# api = tweepy.API(auth)

class QandAListen(tweepy.StreamListener, file):

	def on_status(self,data):

		try:
			file.write(data.text)
		except:
			pass

		try: 
			print data.text
			f = open('tweetdump.txt', 'a')
			f.write(data.text)
			f.write("\n")
			f.close()

		except:
			pass

		return True

	def on_error(self,status_code):
		print 'An error has occured!'
		return True

	def on_timeout(self):
		print 'Snoozing...'

with open('tweetdump', 'w') as f:
	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_TOKEN,ACCESS_TOKEN_SECRET)

	stream = tweepy.Stream(auth, QandAListen(f), timeout=None)
	stream.filter(track=['savetheundies'])