import tweepy
from json import dumps

consumer_key = "***REMOVED***"
consumer_secret = "***REMOVED***"
access_token = "***REMOVED***"
access_token_secret = "***REMOVED***"

condolence_indicators = ['sorry for your loss',
                         'my heart goes out to you',
                         'hope you find peace',
                         'my thoughts and prayers',
                         'my deepest condolences',
                         ]
# condolence_indicators = ['sorry for your loss']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(dumps(status._json))

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
myStream.filter(track=condolence_indicators)