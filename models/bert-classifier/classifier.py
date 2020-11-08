from torch import nn
from pytorch_transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(num_labels)
        nn.init.xavier_normal_(self.classifier.weight)
    
    def forward(self, indexed_toks, apply_softmax=False):
        bert_outputs = self.bert(indexed_toks, token_type_ids=None)
        # Consider this from the docs:
        # This output is usually not a good summary of the semantic content of the
        # input, you’re often better with averaging or pooling the sequence of
        # hidden-states for the whole input sequence.
        pooled_hidden_state = bert_outputs[1]
        pooled_hidden_state = self.dropout(pooled_hidden_state)
        output = self.classifier(pooled_hidden_state)
        if apply_softmax:
            output = self.softmax(logits, dim=1)
        return output
