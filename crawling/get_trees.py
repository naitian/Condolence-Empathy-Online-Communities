#!env/bin/python

import argparse
import lzma
from sys import stdin

import ujson

def main(args):
    # Create a set of submissions_links
    classified_file = "./parsed/classified_comments/sample_comments_{}.tsv".format(args.month)
    post_tree_file = "/shared/0/datasets/reddit/post-thread-trees/{}.json.xz".format(args.month)
    post_ids = set()
    with open(classified_file, "r") as f:
        for post_id in f:
            post_ids.add(post_id.split("\t")[1][3:])
    trees_out = open(
        "./parsed/classified_comments/comment_trees_{}.tsv".format(args.month), "w"
    )
    for i, tree in enumerate(lzma.open(post_tree_file, "rt")):
        if i % 100000 == 0:
            print(i)
        try:
            tree_id = ujson.loads(tree)["id"]
            if tree_id in post_ids:
                trees_out.write(tree)
        except ValueError:
            continue
    trees_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetches parent comment and submission references"
    )
    parser.add_argument(
        "month", action="store", help="path to file with post id on each line"
    )
#    parser.add_argument(
#        "tree_ids", action="store", help="Path to file with tree id (post) on each line"
#    )
#    parser.add_argument(
#        "trees", action="store", help="Path to file with the tree on each line"
#    )
#    parser.add_argument(
#        "timeframe",
#        action="store",
#        help="YYYY-MM reference to timeframe for comments and submissions",
#    )
    args = parser.parse_args()
    main(args)
