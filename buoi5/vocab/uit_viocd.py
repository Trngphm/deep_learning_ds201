import torch
import torch.nn.utils.rnn as rnn_utils
import json
from collections import Counter
from torch.utils.data import Dataset
import json


def collate_fn(items):
    input_ids = [item["input_ids"] for item in items]
    labels = torch.tensor([item["label"] for item in items], dtype=torch.long)

    input_ids_padded = rnn_utils.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )

    return {
        "input_ids": input_ids_padded,
        "labels": labels
    }

class Vocab:
    def __init__(self, json_file, min_freq=1):
        words_counter = Counter()
        domain_set = set()

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for _, item in data.items():
            words_counter.update(item["review"].lower().split())
            domain_set.add(item["domain"])

        # Special tokens
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.bos = "<s>"

        # Word vocab
        self.word2id = {
            self.pad: 0,
            self.bos: 1,
            self.unk: 2
        }
        self.id2word = {v: k for k, v in self.word2id.items()}

        idx = 3
        for word, freq in words_counter.items():
            if freq >= min_freq:
                self.word2id[word] = idx
                self.id2word[idx] = word
                idx += 1

        # Domain vocab (label classification)
        self.domain2id = {
            d: i for i, d in enumerate(sorted(domain_set))
        }
        self.id2domain = {i: d for d, i in self.domain2id.items()}

    def encode_review(self, review):
        words = review.lower().split()
        return torch.tensor(
            [self.word2id.get(w, self.word2id[self.unk]) for w in words],
            dtype=torch.long
        )

    def encode_domain(self, domain):
        return self.domain2id[domain]

    @property
    def vocab_size(self):
        return len(self.word2id)

    @property
    def n_labels(self):
        return len(self.domain2id)
    

class UITViOCDDataset(Dataset):
    def __init__(self, json_file, vocab=None):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.keys = list(self.data.keys())

        if vocab is None:
            self.vocab = Vocab(json_file)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        item = self.data[self.keys[idx]]

        input_ids = self.vocab.encode_review(item["review"])
        label = self.vocab.encode_domain(item["domain"])

        return {
            "input_ids": input_ids,
            "label": label
        }
