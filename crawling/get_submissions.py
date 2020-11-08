#!env/bin/python

import argparse
from sys import stdin


def main(args):
    # Create a set of submissions_links
    submissions_links = set()
    with open("./parsed/unique_submission_links_{}.csv".format(args.timeframe)) as file:
        for line in file:
            submissions_links.add(line.strip())
    # each line should have:
    # .id,.permalink,.title,.selftext,url,.subreddit,.score,.num_comments
    submissions_out = open("./parsed/parent_submissions_{}.tsv".format(args.timeframe), "w")
    for i, line in enumerate(stdin):
        if i % 100000 == 0:
            print(i)
        items = line.split("\t")
        if items[0] in submissions_links:
            # is a parent
            submissions_out.write(line)
    submissions_out.close()


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
