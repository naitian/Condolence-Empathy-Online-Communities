import argparse
import lzma
import ujson


def main(args):
    labeled_file = "./parsed/classified_comments/sample_comments_{}.tsv".format(
        args.month
    )
    reddit_file = "/shared/2/datasets/reddit-dump-all/RC/RC_{}.xz".format(args.month)
    outfile = open(
        "./parsed/classified_comments/condolence_parents_{}.tsv".format(args.month),
        "w",
    )
    whitelist = set()
    print("Loading in Condolence IDs")
    for line in open(labeled_file, "r"):
        try:
            _, _, parent_id, id, _, _, _, _, _, _, _, score, _ = line.split("\t")
        except:
            continue
        score = float(score)
        if score >= 0.9:
            whitelist.add(parent_id)
    print("Loaded {} IDs into Whitelist".format(len(whitelist)))
    print("Crawling All Reddit Comments")
    for i, line in enumerate(lzma.open(reddit_file, "rt")):
        comment = ujson.loads(line)
        if i % 100000 == 0:
            print(i)
        if "t1_" + comment["id"] in whitelist:
            outfile.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("month", action="store", help="Month to find reponses for")
    main(parser.parse_args())
