import pandas as pd
import numpy as np
from markdown import markdown
from bs4 import BeautifulSoup
import re
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.svm import LinearSVC

import time
import copy

from utils import read_comments, preprocess


def markdown_to_text(markdown_string):
    """
    Converts a markdown string to plaintext
    https://gist.github.com/lorey/eb15a7f3338f959a78cc3661fbc255fe
    """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r"<pre>(.*?)</pre>", " ", html)
    html = re.sub(r"<code>(.*?)</code >", " ", html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = "".join(soup.findAll(text=True))

    return text


def strip_links(text):
    return re.sub(r"https?://\S+", "", text)


def remove_utf(text):
    # gets rid of emojis and other stuff (like lenny faces...)
    return text.encode("ascii", "ignore").decode("ascii")


def fit_svm(X, y):
    vectorizer = CountVectorizer(max_features=50000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(X)
    clf = LinearSVC()
    clf.fit(X, y)
    return clf, vectorizer


def main():
    np.random.seed(0)
    # data = "non_condolence_condolence_seeking_2017"
    data = "condolence_comments_2016"
    pos = read_comments("../parsed/{}.tsv".format(data))
    pos["label"] = 1
    print(pos.shape)
    neg = read_comments("../parsed/neg_{}.tsv".format(data))
    neg["label"] = -1
    print(neg.shape)
    df = (
        pd.concat([pos, neg])
        .dropna(subset=["body"])
        .sample(frac=1)
        .reset_index(drop=True)
    )
    df.body = (
        df.body.apply(markdown_to_text)
        .str.replace("\n", " ")
        .apply(remove_utf)
        .apply(strip_links)
    )

    print("loaded data")
    (X_train, X_test, y_train, y_test) = train_test_split(
        df.body.values.tolist(),
        df.label.values.tolist(),
        test_size=0.2,
    )
    (X_val, X_test, y_val, y_test) = train_test_split(X_test, y_test, test_size=0.5)
    print("split data, fitting svm")
    clf, v = fit_svm(X_train, y_train)
    print("svm fit")
    print("svm val")
    x_val = v.transform(X_val)
    print(clf.score(x_val, y_val))
    print(f1_score(clf.predict(x_val), y_val))
    print("precision:", precision_score(clf.predict(x_val), y_val))
    print("recall:", recall_score(clf.predict(x_val), y_val))
    print("svm test")
    x_test = v.transform(X_test)
    print(clf.score(x_test, y_test))
    print(f1_score(clf.predict(x_test), y_test))
    print("precision:", precision_score(clf.predict(x_test), y_test))
    print("recall:", recall_score(clf.predict(x_test), y_test))


if __name__ == "__main__":
    main()
