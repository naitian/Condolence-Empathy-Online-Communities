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
        yield tweet, score


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
        for i, tweet, score in enumerate(
            get_tweets(
                "../data/classified-tweets/tweets_{}.tsv".format(args.timeframe),
            )
        ):
            if i % 100 == 0:
                print(i)
            outfile.write(tweet + "\t" + score + "\n")
