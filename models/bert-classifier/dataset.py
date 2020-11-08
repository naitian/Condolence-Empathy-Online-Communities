import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class CondolenceDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_seq_length=128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        # Tokenize string
        tokens = self.tokenizer.tokenize(self.X[i])
        tokens = tokens[:self.max_seq_length]
        # Convert tokens to Bert vocab indices
        indexed_toks = self.tokenizer.convert_tokens_to_ids(tokens)
        # Pad up to max_seq_length (default 128)
        indexed_toks += [0] * (self.max_seq_length - len(indexed_toks))
        indexed_toks = torch.tensor(indexed_toks)

        label = torch.tensor(self.y[i])

        return indexed_toks, label
