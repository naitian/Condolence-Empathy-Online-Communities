import argparse
import glob
import lzma
import ujson

REDDIT_PATH = "/shared/2/datasets/reddit-dump-all/RC/RC_{}.xz"


def main(args):
    blacklist = set()
    print("Loading blacklist")
    outfile = open(
        "../data/classified-reddit/blacklists/{}.txt".format(args.month), "w"
    )
    for f in glob.glob(args.old):
        print("= reading {}".format(f))
        with open(f, "r") as bl:
            blacklist = blacklist.union(set(bl.read().splitlines()))
    print("Loaded blacklist")
    with lzma.open(REDDIT_PATH.format(args.month), "r") as month:
        for i, comment in enumerate(month):
            if i % 100000 == 0:
                print(i)
            c = ujson.loads(comment)
            if c["id"] in blacklist:
                outfile.write(c["id"] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split gross lame bad monolithic ID files into good nice files bucketed by month"
    )
    parser.add_argument("month", help="YYYY-MM month buckets to generate")
    parser.add_argument("old", help="Path / glob to old bad files")
    main(parser.parse_args())
