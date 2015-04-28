This package is designed to perform sentiment analysis on the Twitter feed during the broadcast of Q&A, an Australian political panel show which is notorious for its lively Twitter feed during the broadcast. 

`ScrapeStream.py` is an edited version of Tweepy's default live stream scraper for the the Twitter API, which uses a filter of keywords and is adapted to store the stream in a text file (`tweetdump.txt` and similar files). 

`TweetSentimentBasic.py` does a basic first pass of sentiment analysis using the out-of-the-box dictionary `twitter_sentiment_list.csv` which I found online, and using the files `LeftList` and `RightList` that I created to identify entities. 

`TrainSentiments.py` prompts for human classification of tweets as positive, negative, neutral or spam. It is set up to be able to be quit at any time and recontinued, and outputs classified tweets to `trained_tweets.txt` and the full cleaned sentiment dictionary to `SentimentDict.txt` (which is actually a csv). 

The main file, `ClassifyTweets.py`, uses the trained sentiment dictionary to classify tweets as positive or negative, and uses the lists to identify entities. 