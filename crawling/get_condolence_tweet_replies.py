#!env/bin/python

import argparse
import csv
import glob
import gzip
import ujson
import random
from itertools import islice
import re
from sys import stdin

import pandas as pd
import torch
from torch import nn

from bert_classifier.classifier import BertClassifier
from bert_classifier.utils import preprocess
from pytorch_transformers import BertTokenizer

# device = "cuda:4"

THRESHOLD = 0.9
# RES_SAMPLE_SIZE = 6_250_000
RES_SAMPLE_SIZE = 1000000


def load_model(path, device="cuda:4"):
    model = BertClassifier(2)
    if device == "cpu":
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    else:
        model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def reservoir_sample(n, iterable=None):
    sample = []
    counter = 0
    for i, line in enumerate(iterable):
        if i % 100000 == 0:
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


def filter_comments(iterable=None, blacklist="../data/tweet-replies/blacklists/*.txt"):
    """
    Yields comments with the following filtered out:
        - comments from outside the top 10k subreddits
        - removed or deleted comments
        - comments which have already been visited (tracked by the blacklist files)
    """
    ids = set()
    for bl in glob.glob(blacklist):
        with open(bl, "r") as blacklist:
            ids = ids.union(set(blacklist.read().splitlines()))
    filter_re = [re.compile(pat) for pat in ["[0-9]", "http", "#"]]
    for i, line in enumerate(iterable):
        tweet = ujson.loads(line)
        tweet_id = tweet["id_str"]
        body = tweet["text"]
        if (
            True
            # tweet_id[-1] == "0"  # rough 10% sample
            and tweet_id not in ids
            and not any([pat.search(body) for pat in filter_re])
        ):
            ids.add(tweet_id)
            yield body, line, tweet_id


def batch_generator(
    tokenizer,
    batch_size=16,
    iterable=None,
    blacklist="../data/tweet-replies/blacklists/*.txt",
    device="cuda:4",
):
    with open(blacklist, "a+") as blacklist:
        ct = iterable
        piece = islice(ct, batch_size)
        while piece:
            input_list = []
            lines = []
            for body, line, c_id in piece:
                tokens = tokenizer.tokenize(preprocess(body))
                tokens = tokens[:128]
                # Convert tokens to Bert vocab indices
                indexed_toks = tokenizer.convert_tokens_to_ids(tokens)
                # Pad up to max_seq_length (default 128)
                indexed_toks += [0] * (128 - len(indexed_toks))
                input_list.append(torch.tensor(indexed_toks).unsqueeze(0))
                lines.append(line)
                blacklist.write(c_id + "\n")
            if len(input_list) == 0:
                return
            batch_inputs = torch.cat(input_list, 0)
            yield batch_inputs.to(device), lines
            piece = islice(ct, batch_size)


def file_generator(timeframe):
    files = glob.glob("../data/tweet-replies/replies_{}.json".format(timeframe))
    for f in files:
        with open(f) as content:
            for tweet in content:
                yield tweet.strip()


def main(args):
    # each line should look like:
    # .created_utc,.link_id,.parent_id,.id,.permalink,.body,.gilded,.controversiality,.score,.author,.subreddit

    print("Classifying {}".format(args.timeframe))
    print("Using GPU {}".format(args.device))
    print("Loading Tokenizer")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("Loaded Tokenizer")
    ns_seeking = load_model(
        "./bert_classifier/notsorry_seeking_models/best_model_2.pth",
        device=args.device,
    )
    print("Loaded Seeking")
    condolence = load_model(
        "./bert_classifier/condolence_models/best_model_2.pth", device=args.device
    )
    print("Loaded Condolence")
    print("Loaded Models")
    if args.sample:
        outfile = open(
            "../data/tweet-replies/labeled_{}.tsv".format(args.timeframe), "a",
        )
    else:
        outfile = open(
            "../data/tweet-replies/labeled_{}.tsv".format(args.timeframe), "w"
        )
    print("Opened file")

    tweet_generator = file_generator(args.timeframe)
    comment_generator = filter_comments(
        iterable=tweet_generator,
        blacklist="../data/tweet-replies/blacklists/{}.txt".format(args.timeframe),
    )
    comment_src = (
        iter(reservoir_sample(n=RES_SAMPLE_SIZE, iterable=comment_generator))
        if args.sample
        else comment_generator
    )
    print(type(comment_src))
    for i, (batch_inputs, lines) in enumerate(
        batch_generator(
            tokenizer,
            batch_size=16,
            iterable=comment_src,
            blacklist="../data/tweet-replies/blacklists/{}.txt".format(args.timeframe),
            device=args.device,
        )
    ):
        condolence_score = nn.functional.softmax(condolence(batch_inputs), dim=1)[:, 1]
        ns_seeking_score = nn.functional.softmax(ns_seeking(batch_inputs), dim=1)[:, 1]
        scores = torch.cat(
            [condolence_score.unsqueeze(1), ns_seeking_score.unsqueeze(1)], 1
        )
        keep = torch.max(scores, dim=1).values > THRESHOLD
        keep_bools = keep.cpu().numpy().astype(bool)

        comments = pd.DataFrame(lines)
        keep_scores = scores[keep].cpu().detach().numpy()
        keep_rows = comments[keep_bools]

        keep_df = pd.concat(
            [comments[keep_bools].reset_index(drop=True), pd.DataFrame(keep_scores)],
            axis=1,
            ignore_index=True,
        )
        print(args.timeframe, i)
        print(ujson.loads(lines[0])["text"])
        print(scores[0])
        outfile.write(
            keep_df.to_csv(sep="\t", header=False, index=False, quoting=csv.QUOTE_NONE)
        )
        del condolence_score
        del ns_seeking_score
        del scores
        del keep
        del keep_bools
        del keep_scores
        del keep_rows
    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetches parent comment and submission references"
    )
    parser.add_argument(
        "timeframe",
        action="store",
        help="YYYY-MM or YYYY-MM-DD reference to timeframe for comments and submissions",
    )
    parser.add_argument(
        "device", action="store", help="CUDA device",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use reservoir sampling to only process a random subset of comments",
    )
    main(parser.parse_args())
