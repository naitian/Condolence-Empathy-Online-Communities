import pandas as pd
import numpy as np
from markdown import markdown
from bs4 import BeautifulSoup
import re
import json

from pytorch_transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
import torch
from torch.utils.data import DataLoader
from torch import optim, nn

import time
import copy

from dataset import CondolenceDataset
from classifier import BertClassifier
from utils import read_comments, preprocess


def test_model(model, data, device):
    dataloaders, dataset_sizes = data
    print("Testing model...")
    start = time.time()
    model.eval()
    num_correct = 0
    predicted = []
    actual = []
    flag = False
    for reviews, sentiments in dataloaders["val"]:
        reviews = reviews.to(device)
        sentiments = sentiments.to(device)
        predictions = nn.functional.softmax(model(reviews))
        if not flag:
            print(predictions)
            flag = True
        predicted += list(torch.max(predictions, 1)[1].cpu().numpy())
#         predicted += list((predictions.detach().cpu().numpy()[:,1] > 0.9).astype(int))
        actual += list(torch.max(sentiments, 1)[1].cpu().numpy())
        num_correct += torch.sum(
            torch.max(predictions, 1)[1] == torch.max(sentiments, 1)[1]
        )
#        num_correct += torch.sum(predictions[:,1] > 0.9)
    end = time.time()
    print(
        "Tested Model in {:0f}m {:0f}s".format((end - start) // 60, (end - start) % 60)
    )
    print(num_correct)
    print(predicted[:10])
    print(actual[:10])
    print(dataset_sizes["test"])
    print("Test accuracy of {}".format(num_correct.cpu().numpy() / dataset_sizes["test"]))
    print("F1 Score of {}".format(f1_score(actual, predicted)))
    print("Precision Score of {}".format(precision_score(actual, predicted)))
    print("Recall Score of {}".format(recall_score(actual, predicted)))
    print("Classification Report:")
    print(classification_report(actual, predicted, target_names=["False", "True"]))

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


def main():
    np.random.seed(0)
    # data = "non_condolence_condolence_seeking_2017"
    data = "condolence_comments_2016"
    pos = read_comments("../parsed/{}.tsv".format(data))
    pos['label'] = ['positive'] * pos.shape[0]
    print(pos.shape)
    neg = read_comments("../parsed/neg_{}.tsv".format(data))
    neg['label'] = ['negative'] * neg.shape[0]
    print(neg.shape)
    df = pd.concat([pos, neg]).dropna(subset=["body"]).sample(frac=1).reset_index(drop=True)
    df.body = df.body.apply(markdown_to_text).str.replace("\n", " ").apply(remove_utf).apply(strip_links)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    (X_train, X_test, y_train, y_test) = train_test_split(
        df.body.values.tolist(),
        pd.get_dummies(df.label).values.tolist(),
        test_size=0.2,
    )
    (X_val, X_test, y_val, y_test) = train_test_split(X_test, y_test, test_size=0.5)


    train_data = CondolenceDataset(X_train, y_train, tokenizer)
    val_data = CondolenceDataset(X_val, y_val, tokenizer)
    test_data = CondolenceDataset(X_test, y_test, tokenizer)

    print("Loading Model")
    classifier = BertClassifier(2)
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    classifier.load_state_dict(torch.load("./bert_classifier/condolence_models_retrain/best_model_1_retrain.pth", map_location=device))
    # classifier.load_state_dict(torch.load("./bert_classifier/notsorry_seeking_models_retrain/best_model_1_retrain.pth", map_location=device))
    classifier.to(device)
    print("Loaded Model")


    dataloaders = {
        "train": DataLoader(train_data, batch_size=16, num_workers=0),
        "val": DataLoader(val_data, batch_size=16, num_workers=0),
        "test": DataLoader(test_data, batch_size=16, num_workers=0),
    }

    dataset_sizes = {"train": 0, "val": 0, "test": len(test_data)}

    data = (dataloaders, dataset_sizes)

    test_model(classifier, data, device)

   
if __name__ == "__main__":
    main()
