import pandas as pd
import numpy as np
from markdown import markdown
from bs4 import BeautifulSoup
import re
import json

from pytorch_transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch import optim, nn

import time
import copy

from dataset import CondolenceDataset
from classifier import BertClassifier
from utils import read_comments, preprocess


def train_model(model, data, optimizer, loss_fn, scheduler, device, num_epochs=25):
    dataloaders, dataset_sizes = data
    print("Training model...")
    start = time.time()
    training_state = {
        "train": {"loss": [], "accuracy": []},
        "val": {"loss": [], "accuracy": []},
        "time_start": start,
        "time_end": None,
        "time_epoch_start": [],
        "time_epoch_end": [],
    }
    for epoch in range(num_epochs):
        training_state["time_epoch_start"].append(time.time())
        for phase in ["train", "val"]:
            running_loss = 0
            num_correct = 0
            if phase == "train":
                model.train()
            else:
                model.eval()
            for reviews, sentiments in dataloaders[phase]:
                reviews = reviews.to(device)
                sentiments = sentiments.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    predictions = model(reviews)
                    loss = loss_fn(predictions, torch.max(sentiments.float(), 1)[1])
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * reviews.size(0)
                num_correct += torch.sum(
                    torch.max(predictions, 1)[1] == torch.max(sentiments, 1)[1]
                )

            accuracy = num_correct.double() / dataset_sizes[phase]
            epoch_loss = running_loss / dataset_sizes[phase]
            print("{} loss: {:4f}".format(phase, epoch_loss))
            print("{} accuracy: {:4f}".format(phase, accuracy))
            if phase == "val" and epoch_loss < min(
                training_state["val"]["loss"] + [100]
            ):
                torch.save(model.state_dict(), "./notsorry_seeking_models_retrain/best_model_{}_retrain.pth".format(epoch))
            training_state[phase]["loss"].append(epoch_loss)
            training_state[phase]["accuracy"].append(accuracy.cpu().detach().numpy().tolist())
        with open("epoch_{}_state.json".format(epoch), "w") as f:
            f.write(json.dumps(training_state))
        scheduler.step()
        training_state["time_epoch_end"].append(time.time())

    end = time.time()
    training_state["time_end"] = end
    print(
        "Trained Model in {:0f}m {:0f}s".format((end - start) // 60, (end - start) % 60)
    )
    return training_state

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
    data = "non_condolence_condolence_seeking_2017"
    # data = "condolence_comments_2016"
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

    classifier = BertClassifier(2)
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    classifier.to(device)

    dataloaders = {
        "train": DataLoader(train_data, batch_size=16, num_workers=0, shuffle=True),
        "val": DataLoader(val_data, batch_size=16, num_workers=0),
        "test": DataLoader(test_data, batch_size=16, num_workers=0),
    }

    dataset_sizes = {"train": len(train_data), "val": len(val_data), "test": len(test_data)}

    data = (dataloaders, dataset_sizes)

    optimizer = optim.Adam(
        [
            {"params": classifier.bert.parameters(), "lr": 0.00001},
            {"params": classifier.classifier.parameters(), "lr": 0.001},
        ]
    )

    loss_fn = nn.CrossEntropyLoss()
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model, training_state = train_model(
        classifier, data, optimizer, loss_fn, exp_lr_scheduler, device
    )

   
if __name__ == "__main__":
    main()
