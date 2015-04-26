import time
from getpass import getpass
from textwrap import TextWrapper

import tweepy

from keys import keys

consumer_key = keys['consumer_key']
consumer_secret = keys['consumer_secret']
access_token = keys['access_token']
access_token_secret = keys['access_token_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

search_text = "#qanda"
search_number = 1000
search_result = api.search(search_text, rpp=search_number)
for i in search_result:
    try:
    	print i.text
    	f = open('tweetdump.txt', 'a')
    	f.write(i.text)
    	f.write("\n")

    except:
    	pass


