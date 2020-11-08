#!env/bin/python

import argparse
from sys import stdin


def main(args):
    # Create a set of parent_ids
    parent_ids = set()
    with open("./parsed/unique_parent_ids_{}.csv".format(args.timeframe)) as file:
        for line in file:
            parent_ids.add(line.strip())

    # each line should look like:
    # .link_id,.parent_id,.id,.permalink,.body,.gilded,.controversiality,.score,.author,.subreddit
    parent_out = open("./parsed/parent_comments_{}.tsv".format(args.timeframe), "w")
    child_out = open("./parsed/children_comments_{}.tsv".format(args.timeframe), "w")
    for i, line in enumerate(stdin):
        if i % 100000 == 0:
            print(i)
        items = line.split("\t")
        if items[1] in parent_ids:
            # is a child
            child_out.write(line)
        if "t1_" + items[2] in parent_ids:
            # is a parent
            parent_out.write(line)
    parent_out.close()
    child_out.close()


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
