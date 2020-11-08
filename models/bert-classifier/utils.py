import pandas as pd
from markdown import markdown
from bs4 import BeautifulSoup
import re


def read_comments(fname, timestamp=False, **kwargs):
    print("Reading comments from file {}".format(fname))
    headers = [
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
        ]
    if timestamp:
        headers = ["created_utc"] + headers
        print(headers)
    comments = pd.read_csv(
        fname,
        sep="\t",
        names=headers,
        **kwargs,
    ).dropna(subset=["body"])
    return comments

def markdown_to_text(markdown_string):
    """
    Converts a markdown string to plaintext
    https://gist.github.com/lorey/eb15a7f3338f959a78cc3661fbc255fe
    """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text

def strip_links(text):
    return re.sub(r'https?://\S+', '', text)

def remove_utf(text):
    # gets rid of emojis and other stuff (like lenny faces...)
    return text.encode('ascii', 'ignore').decode('ascii')

def preprocess(text):
    return strip_links(remove_utf(markdown_to_text(text).replace("\n", " ")))