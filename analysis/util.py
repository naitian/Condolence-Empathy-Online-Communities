import pandas as pd


def read_labeled_output(fname):
    print("Loading labeled comments from {}".format(fname))
    df = pd.read_csv(fname,
                   sep="\t",
                   names=[
                       "created_utc",
                       "link_id",
                       "parent_id",
                       "id",
                       "permalink",
                       "body",
                       "gilded",
                       "controversiality",
                       "score",
                       "author",
                       "subreddit",
                       "condolence_score",
                       "seeking_score"
                   ])
    df["created_utc"] = pd.to_datetime(df.created_utc, unit="s", utc=True)
    df["is_c"] = df.condolence_score >= 0.9
    df["is_cs"] = df.seeking_score >= 0.9
    df["is_both"] = df.is_c & df.is_cs
    return df, df[df.is_c], df[df.is_cs], df[df.is_both]
