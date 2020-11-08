#!env/bin/python

import argparse
import re
from sys import stdin


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
    phrases = ["sorry for your loss"] if args.bootstrap else condolence_phrases
    # each line should look like:
    # .link_id,.parent_id,.id,.permalink,.body,.gilded,.controversiality,.score,.author,.subreddit
    parent_ids = open("./parsed/parent_ids_{}.csv".format(args.timeframe), "w")
    submission_links = open(
        "./parsed/submission_links_{}.csv".format(args.timeframe), "w"
    )
    for i, line in enumerate(stdin):
        if i % 100000 == 0:
            print(i)
        items = line.split("\t")
        if any(phrase in items[4].lower() for phrase in phrases):
            if items[1][:3] == "t3_":
                submission_links.write(items[1][3:] + "\n")
            parent_ids.write(items[1] + "\n")


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
        "-b",
        "--bootstrap",
        action="store_true",
        dest="bootstrap",
        help="use only 'sorry for your loss' as a seed to discover other condolence phrases",
    )
    args = parser.parse_args()
    main(args)
