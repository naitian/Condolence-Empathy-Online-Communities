#!env/bin/python

import argparse
import random
from sys import stdin
from collections import defaultdict


condolence_phrases = [
    "made me tear up",
    "you dodged a bullet",
    "take care of yourself",
    "even begin to imagine",
    "my heart goes out",
    "not beat yourself up",
    "please take care of",
    "keep your head up",
    "heart goes out to",
    "can not even begin",
    "do not blame yourself",
    "hope you find peace",
    "my thoughts and prayers",
    "there are no words",
    "this made me cry",
    "remember the good times",
    "my deepest condolences",
    "can not imagine losing",
    "can not even imagine",
    "god bless you and",
    "sorry for your loss",  # added manually
]


def main(args):
    # each line should look like:
    # .link_id,.parent_id,.id,.permalink,.body,.gilded,.controversiality,.score,.author,.subreddit

    filtered_out = open(
        "./parsed/filtered_parent_comments_{}.tsv".format(args.timeframe), "w"
    )
    f = open("./parsed/parent_comments_{}.tsv".format(args.timeframe))
    print("./parsed/parent_comments_{}.tsv".format(args.timeframe))
    print("Opened file")
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print(i)
        _, parent_id, _, _, body, _, _, _, _, _ = line.split("\t")
        if (
            any(phrase in body.lower() for phrase in ["[deleted]", "[removed]"])
            or len(body) == 0
        ):
            continue
        else:
            filtered_out.write(line)
    filtered_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetches parent comment and submission references"
    )
    parser.add_argument(
        "timeframe",
        action="store",
        help="YYYY-MM reference to timeframe for comments and submissions",
    )
    args = parser.parse_args()
    main(args)
