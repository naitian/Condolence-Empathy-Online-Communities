#!env/bin/python

import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from pytorch_transformers import BertTokenizer
import pickle as pkl

from bert_classifier.classifier import BertClassifier
from bert_classifier.dataset import CondolenceDataset


def read_comments(fname):
    comments = pd.read_csv(
        fname,
        sep="\t",
        names=[
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
        ],
    ).dropna(subset=["body"])
    return comments


def main(args):
    device = "cuda:4"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    condolence_model = BertClassifier(2)
    condolence_model.load_state_dict(torch.load("./bert_classifier/condolence_models/best_model_2.pth"))
    condolence_model.to(device)
    condolence_model.eval()
    seeking_comments = read_comments(args.input)
    comment_text = seeking_comments.body.tolist()
    condolence_score = []
    for i, c in enumerate(comment_text):
        tokens = tokenizer.tokenize(c)
        tokens = tokens[:128]
        # Convert tokens to Bert vocab indices
        indexed_toks = tokenizer.convert_tokens_to_ids(tokens)
        # Pad up to max_seq_length (default 128)
        indexed_toks += [0] * (128 - len(indexed_toks))
        indexed_toks = torch.tensor(indexed_toks).to(device).unsqueeze(0)
        score = nn.functional.softmax(condolence_model(indexed_toks), dim=1)
        condolence_score.append(float(score[0][1]))
        del tokens
        del indexed_toks
        del score
        if i % 1000 == 0:
            print(c)
            print(condolence_score[-1])
    seeking_comments['condolence_score'] = condolence_score
    pkl.dump(condolence_score, open("./parsed/condolence_scores_seeking_2017.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
       description="Runs condolence classifier and gets scores for all comments in input file"
    )
    parser.add_argument(
        "input",
        action="store",
        help="Path to input file",
    )
    args = parser.parse_args()
    main(args)
