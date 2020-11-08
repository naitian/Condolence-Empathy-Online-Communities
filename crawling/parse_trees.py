#!env/bin/python

import argparse
from ujson import loads
from sys import stdin
from collections import deque


def main(args):
    with open(args.comment_ids, "r") as c_ids:
        comment_ids = set([c_id.strip() for c_id in c_ids])

    trees_out = open(
        "./parsed/classified_comments/flattened_comment_trees_{}.tsv".format(args.timeframe), "w"
    )
    trees = open(args.trees, "r")
    for i, tree in enumerate(trees):
        try:
            tree = loads(tree)
        except:
            print("JSON loads ran into exception decoding tree {}".format(i))
            continue
        if i % 10000 == 0:
            print(i)
        comments = deque(tree["children"])
        parents = set()
        while len(comments) > 0:
            c = comments.popleft()
            comments.extend(c["children"])
            if c["id"] in comment_ids or c["parent_id"][3:] in parents:
                parents.add(c["id"])
                trees_out.write("\t".join([c["id"]] + [x["id"] for x in c["children"]]) + "\n")
    trees_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetches parent comment and submission references"
    )
    parser.add_argument(
        "trees", action="store", help="path to extracted comment trees file"
    )
    parser.add_argument(
        "comment_ids", action="store", help="list of comment ids to keep"
    )
    parser.add_argument(
        "timeframe",
        action="store",
        help="YYYY-MM reference to timeframe for comments and submissions",
    )
    args = parser.parse_args()
    main(args)
