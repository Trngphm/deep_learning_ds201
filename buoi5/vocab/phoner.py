import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import json
from collections import Counter

# Collate function cho DataLoader
def collate_fn(items):
    input_ids = [item["input_ids"] for item in items]
    label_ids = [item["labels"] for item in items]

    # Padding input_ids
    input_ids_padded = rnn_utils.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )
    
    # Padding labels
    label_ids_padded = rnn_utils.pad_sequence(
        label_ids,
        batch_first=True,
        padding_value=-100  # dùng -100 cho CrossEntropyLoss ignore_index
    )

    return {
        "input_ids": input_ids_padded,
        "labels": label_ids_padded
    }

# Vocab class cho words + tags
class Vocab:
    def __init__(self, json_file, min_freq=1):
        words_counter = Counter()
        tags_set = set()

        # Load dữ liệu
        with open(json_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        for item in data:
            words_counter.update(item["words"])
            tags_set.update(item["tags"])

        # Special tokens
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.bos = "<s>"

        # Word vocab
        self.word2id = {self.pad:0, self.bos:1, self.unk:2}
        self.id2word = {0:self.pad, 1:self.bos, 2:self.unk}
        idx = 3
        for word, freq in words_counter.items():
            if freq >= min_freq:
                self.word2id[word] = idx
                self.id2word[idx] = word
                idx += 1

        # Tag vocab
        self.tag2id = {tag: i for i, tag in enumerate(sorted(tags_set))}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

    # Encode words và tags
    def encode_words(self, words):
        return torch.tensor([self.word2id.get(w, self.word2id[self.unk]) for w in words], dtype=torch.long)

    def encode_tags(self, tags):
        return torch.tensor([self.tag2id[t] for t in tags], dtype=torch.long)

    @property
    def vocab_size(self):
        return len(self.word2id)

    @property
    def n_labels(self):
        return len(self.tag2id)


# Dataset cho PhoNER
class PhoNERDataset(Dataset):
    def __init__(self, json_file, vocab=None):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        
        if vocab is None:
            self.vocab = Vocab(json_file)
        else:
            self.vocab = vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.vocab.encode_words(item["words"])
        labels = self.vocab.encode_tags(item["tags"])
        return {
            "input_ids": input_ids,
            "labels": labels
        }
