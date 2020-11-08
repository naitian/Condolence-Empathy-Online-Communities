#!env/bin/python

import argparse
import glob
import random
from itertools import islice
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
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model


def reservoir_sample(n, iterable=None):
    sample = []
    counter = 0
    for i, line in enumerate(iterable):
        # process a negative child example
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


def filter_comments(
    whitelist,
    tree_file,
    iterable=None,
    blacklist="./parsed/classified_comments/visited*.txt",
):
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
    with open(tree_file, "r") as tree_list:
        keep = set(tree_list.read().splitlines())
    filters = set(["[removed]", "[deleted]"])
    for i, line in enumerate(iterable):
        # if i % 10000 == 0:
        #     print(i, len(whitelist))
        (_, _, _, comment_id, _, body, _, _, _, _, subreddit) = line.strip().split("\t")
        if len(whitelist) == 0:
            return
        if (
            comment_id in keep
            and subreddit in whitelist
            and body not in filters
            and comment_id not in ids
            and whitelist[subreddit] < 100000
        ):
            ids.add(comment_id)
            yield body, line, comment_id


def batch_generator(
    tokenizer,
    batch_size=16,
    iterable=None,
    blacklist="./parsed/classified_comments/visited.txt",
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
                lines.append(line.strip().split("\t"))
                blacklist.write(c_id + "\n")
            if len(input_list) == 0:
                return
            batch_inputs = torch.cat(input_list, 0)
            yield batch_inputs.to(device), lines
            piece = islice(ct, batch_size)


def main(args):
    # each line should look like:
    # .created_utc,.link_id,.parent_id,.id,.permalink,.body,.gilded,.controversiality,.score,.author,.subreddit

    print("Loading top 10k subreddits")
    whitelist = {
        sub: 0 for sub in pd.read_csv("./parsed/10k_by_comment_count.csv")["subreddit"]
    }
    print("Loaded top 10k subreddits")
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
    outfile = open(
        "./parsed/classified_comments/sample_comments_{}".format(
            args.timeframe
        ),
        "a",
    )
    print("Opened file")

    tree_file = "./parsed/classified_comments/flattened_comment_trees_{}.tsv".format(
        args.timeframe
    )
    comment_generator = filter_comments(
        whitelist=whitelist, tree_file=tree_file, iterable=stdin
    )
    comment_src = comment_generator

    print(type(comment_src) )
    for i, (batch_inputs, lines) in enumerate(
        batch_generator(
            tokenizer,
            batch_size=16,
            iterable=comment_src,
            blacklist="./parsed/classified_comments/visited_{}.txt".format(args.device),
            device=args.device,
        )
    ):
        if i % 50 == 0:
            print(args.timeframe, i)
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
        for sub in keep_df[9]:
            # iterate through subreddits
            if sub in whitelist:
                whitelist[sub] += 1
                if whitelist[sub] > 100000:
                    print("filled {}".format(sub))
                    del whitelist[sub]
        outfile.write(keep_df.to_csv(sep="\t", header=False, index=False))
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
        help="YYYY-MM reference to timeframe for comments and submissions",
    )
    parser.add_argument(
        "device", action="store", help="CUDA device",
    )
    main(parser.parse_args())
