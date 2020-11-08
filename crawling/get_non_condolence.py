#!env/bin/python

import argparse
import re
from collections import defaultdict
from random import randint
from sys import stdin

CHILDREN = "/shared-1/projects/condolence/working-dir/parsed/children_comments_{}.tsv"
FILTERED = (
    "/shared-1/projects/condolence/working-dir/parsed/filtered_children_comments_{}.tsv"
)
PARENTS = "/shared-1/projects/condolence/working-dir/parsed/parent_comments_{}.tsv"
FILTERED_PARENTS = "/shared-1/projects/condolence/working-dir/parsed/filtered_parent_comments_{}.tsv"


def _get_ids_and_sample_cts(fname, use_parent=False):
    comment_ids = set()
    sample_size = defaultdict(int)
    with open(fname, mode="r") as f:
        for line in f:
            (
                _,
                parent_id,
                comment_id,
                _,
                _,
                _,
                _,
                _,
                _,
                subreddit,
            ) = line.strip().split("\t")
            if use_parent:
                comment_ids.add(parent_id)
            else:
                comment_ids.add(comment_id)
            sample_size[subreddit] += 1
    return comment_ids, sample_size


def main(args):
    if args.filtered:
        child_ids, child_size = _get_ids_and_sample_cts(
            FILTERED.format(args.timeframe), use_parent=True
        )
        parent_ids, parent_size = _get_ids_and_sample_cts(FILTERED_PARENTS.format(args.timeframe))
    else:
        child_ids, child_size = _get_ids_and_sample_cts(
            CHILDREN.format(args.timeframe), use_parent=True
        )
        parent_ids, parent_size = _get_ids_and_sample_cts(PARENTS.format(args.timeframe))
    child_samples = defaultdict(list)
    child_num_encountered = defaultdict(int)
    parent_samples = defaultdict(list)
    parent_num_encountered = defaultdict(int)
    for i, line in enumerate(stdin):
        if i % 100000 == 0:
            print(i)
        (_, _, comment_id, _, body, _, _, _, _, subreddit) = line.strip().split("\t")

        if (
            any(phrase in body.lower() for phrase in ["[deleted]", "[removed]"])
            or len(body) == 0
        ):
            continue
        if comment_id not in child_ids and subreddit in child_size:
            # process a negative child example
            child_num_encountered[subreddit] += 1
            if child_num_encountered[subreddit] <= child_size[subreddit]:
                # if i <= n_k, add to sample
                child_samples[subreddit].append(line)
            else:
                # else, keep all old items with probability 1 - (n_k / i)
                # or swap new item out w/ random old item with probability (n_k / i)
                r = randint(0, child_num_encountered[subreddit])
                if r < child_size[subreddit]:
                    child_samples[subreddit][r] = line
        if comment_id not in parent_ids and subreddit in parent_size:
            # process a negative parent example
            parent_num_encountered[subreddit] += 1
            if parent_num_encountered[subreddit] <= parent_size[subreddit]:
                # if i <= n_k, add to sample
                parent_samples[subreddit].append(line)
            else:
                # else, keep all old items with probability 1 - (n_k / i)
                # or swap new item out w/ random old item with probability (n_k / i)
                r = randint(0, parent_num_encountered[subreddit])
                if r < parent_size[subreddit]:
                    parent_samples[subreddit][r] = line
    timeframe = (
        "filtered_{}".format(args.timeframe) if args.filtered else args.timeframe
    )
    parent_out = open("./parsed/neg_parent_comments_{}.tsv".format(timeframe), "w")
    child_out = open("./parsed/neg_children_comments_{}.tsv".format(timeframe), "w")
    print(len(child_samples))
    print(len(parent_samples))
    for sample in child_samples:
        for line in child_samples[sample]:
            child_out.write(line)
    for sample in parent_samples:
        for line in parent_samples[sample]:
            parent_out.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetches parent comment and submission references"
    )
    parser.add_argument(
        "timeframe",
        action="store",
        help="YYYY-MM reference to timeframe for comments and submissions",
    )
    parser.add_argument("--filtered", action="store_true", help="Use filtered comments")
    args = parser.parse_args()
    main(args)
