import torch 
import json
from torch.nn.utils.rnn import pad_sequence
import string

# -------------------------
# Collate function
# -------------------------
def collate_fn(items):
    input_ids = [item["input_ids"] for item in items]
    label_ids = [item["label"] for item in items]
    
    input_ids_padded = pad_sequence(
        input_ids, 
        batch_first=True, 
        padding_value=0
    )
    
    labels_stacked = torch.stack(label_ids)
    
    return {
        "input_ids": input_ids_padded,
        "label": labels_stacked
    }

# -------------------------
# Vocabulary
# -------------------------
class Vocab:
    def __init__(self, json_file):
        all_words = set()
        labels = set()

        # Chỉ load 1 file duy nhất
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build vocab
        for item in data:
            sentence = self.preprocess_sentence(item["sentence"])
            all_words.update(sentence.split())
            labels.add(item["sentiment"])
        
        self.pad = "<pad>"
        self.bos = "<s>"
        self.unk = "<unk>"

        self.w2i = {
            word: idx for idx, word in enumerate(sorted(all_words), start=3)
        }
        self.w2i[self.pad] = 0
        self.w2i[self.bos] = 1
        self.w2i[self.unk] = 2

        self.i2w = {idx: word for word, idx in self.w2i.items()}

        self.label2id = {label: idx for idx, label in enumerate(sorted(labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def preprocess_sentence(self, sentence: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        sentence = sentence.lower()
        sentence = sentence.translate(translator)
        return sentence

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def n_labels(self):
        return len(self.label2id)

    def encode_sentence(self, sentence: str):
        sentence = self.preprocess_sentence(sentence)
        words = [self.bos] + sentence.split()
        word_ids = [self.w2i.get(w, self.w2i[self.unk]) for w in words]
        return torch.tensor(word_ids).long()

    def encode_label(self, label: str):
        return torch.tensor(self.label2id[label]).long()


# -------------------------
# Dataset
# -------------------------
class UITVSFCDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, vocab=None):
        # Load đúng 1 file JSON
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Nếu không có vocab, tạo từ file train
        if vocab is None:
            self.vocab = Vocab(json_file)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.vocab.encode_sentence(item["sentence"])
        label = self.vocab.encode_label(item["sentiment"])

        return {
            "input_ids": input_ids,
            "label": label
        }
