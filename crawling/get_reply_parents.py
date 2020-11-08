"""

Twitter's API doesn't allow you to get replies to a particular tweet. Strange
but true. But you can use Twitter's Search API to search for tweets that are
directed at a particular user, and then search through the results to see if 
any are replies to a given tweet. You probably are also interested in the
replies to any replies as well, so the process is recursive. The big caveat 
here is that the search API only returns results for the last 7 days. So 
you'll want to run this sooner rather than later.

replies.py will read a line oriented JSON file of tweets and look for replies
using the above heuristic. Any replies that are discovered will be written as
line oriented JSON to stdout:

    ./replies.py tweets.json > replies.json

It also writes a log to replies.log if you are curious what it is doing...which
can be handy since it will sleep for periods of time to work within the 
Twitter API quotas.

PS. you'll need to:

    pip install python-twitter

and then set the following environment variables for it to work:

  - CONSUMER_KEY
  - CONSUMER_SECRET
  - ACCESS_TOKEN
  - ACCESS_TOKEN_SECRET

"""

import argparse
import random
import gzip
import sys
import time
import logging
import twitter
import urllib.parse

import ujson as json

random.seed(0)

e = {
    "CONSUMER_KEY": "***REMOVED***",
    "CONSUMER_SECRET": "***REMOVED***",
    "ACCESS_TOKEN": "***REMOVED***",
    "ACCESS_TOKEN_SECRET": "***REMOVED***",
}

t = twitter.Api(
    consumer_key=e["CONSUMER_KEY"],
    consumer_secret=e["CONSUMER_SECRET"],
    access_token_key=e["ACCESS_TOKEN"],
    access_token_secret=e["ACCESS_TOKEN_SECRET"],
    sleep_on_rate_limit=True,
)


def reservoir_sample(n, iterable=None):
    sample = []
    counter = 0
    for i, line in enumerate(iterable):
        if i % 10000 == 0:
            print("Res Sample: processed {} lines".format(i))
        counter += 1
        if counter + 1 <= n:
            # if i <= n_k, add to sample
            sample.append(line)
        else:
            # else, keep all old items with probability 1 - (n_k / i)
            # or swap new item out w/ random old item with probability (n_k / i)
            r = random.randint(0, counter)
            if r < len(sample):
                sample[r] = line
    print("Sampled {} items from {} items".format(len(sample), counter))
    return sample


def tweet_url(t):
    return "https://twitter.com/%s/status/%s" % (t.user.screen_name, t.id)


def get_tweets(filename):
    sample = reservoir_sample(10000, iterable=open(filename))
    for line in sample:
        # tweet = line
        tweet, score = line.split("\t")
        yield twitter.Status.NewFromJsonDict(json.loads(tweet))


def get_replies(tweet):
    user = tweet.user.screen_name
    tweet_id = tweet.id
    max_id = None
    logging.info("looking for replies to: %s" % tweet_url(tweet))
    while True:
        q = urllib.parse.urlencode({"q": "to:%s" % user})
        try:
            replies = t.GetSearch(
                raw_query=q, since_id=tweet_id, max_id=max_id, count=100
            )
        except twitter.error.TwitterError as e:
            logging.error("caught twitter api error: %s", e)
            logging.error(q)
            break
        for reply in replies:
            logging.info("examining: %s" % tweet_url(reply))
            if reply.in_reply_to_status_id == tweet_id:
                logging.info("found reply: %s" % tweet_url(reply))
                yield reply
                # recursive magic to also get the replies to this reply
                for reply_to_reply in get_replies(reply):
                    yield reply_to_reply
            max_id = reply.id
        if len(replies) != 100:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetches parent comment and submission references"
    )
    parser.add_argument(
        "timeframe",
        action="store",
        help="YYYY-MM or YYYY-MM-DD reference to timeframe for comments and submissions",
    )
    args = parser.parse_args()
    with open(
        "../data/tweet-replies/parents_{}.json".format(args.timeframe), "wt"
    ) as outfile:
        logging.basicConfig(filename="replies.log", level=logging.INFO)
        tweets_file = sys.argv[1]
        for parent in get_tweets(
                "../data/classified-tweets/tweets_{}.tsv".format(args.timeframe),
                ):
            outfile.write(json.dumps(parent) + "\n")
