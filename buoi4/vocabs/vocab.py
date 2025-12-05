import json
import os
import re
import torch

class Vocab:
    def __init__(self, path, src_lang, tgt_lang):
        self.initialize_vocab(path, src_lang, tgt_lang)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
    def initialize_vocab(self, path, src_lang, tgt_lang):
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        
        self.specials = [self.bos_token, self.eos_token, self.pad_token, self.unk_token]
        
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        
    def make_vocab(self, path, src_lang, tgt_lang):
        json_file = os.listdir(path)
        src_words = set()
        tgt_words = set()
        
        self.max_sentence_length = 0
        
        for json_f in json_file:
            data = json.load(open(os.path.join(path, json_f), 'r', encoding='utf-8'))
            
            for item in data:
                src_sentence = item[src_lang]
                tgt_sentence = item[tgt_lang]
                
                src_tokens =  self.preprocess_sentence(src_sentence)
                tgt_tokens =  self.preprocess_sentence(tgt_sentence)
                
                src_words.update(src_tokens)
                tgt_words.update(tgt_tokens)
                
                if self.max_sentence_length < len(tgt_sentence):
                    self.max_sentence_length = len(tgt_sentence)
                
        src_itos = self.specials + sorted(list(src_words))
        self.src_itos = {i: tok for i, tok in enumerate(src_itos)}
        self.src_stoi = {tok: i for i, tok in enumerate(src_itos)}
        
        tgt_itos = self.specials + sorted(list(tgt_words))
        self.tgt_itos = {i: tok for i, tok in enumerate(tgt_itos)}
        self.tgt_stoi = {tok: i for i, tok in enumerate(tgt_itos)}
        
    @property
    def total_src_tokens(self):
        return len(self.src_itos)
    
    @property
    def total_tgt_tokens(self):
        return len(self.tgt_itos)
    
    def preprocess_sentence(self, sentence):
        sentence = sentence.lower().strip()
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = re.sub(r"!", " ! ", sentence)
        sentence = re.sub(r"\?", " ? ", sentence)
        sentence = re.sub(r"\"", " \" ", sentence)
        sentence = re.sub(r",", " , ", sentence)
        sentence = re.sub(r":", " : ", sentence)
        sentence = re.sub(r";", " ; ", sentence)
        sentence = re.sub(r"\(", " ( ", sentence)
        sentence = re.sub(r"\)", " ) ", sentence)
        sentence = re.sub(r"\.", " . ", sentence)
        sentence = re.sub(r"'", " ' ", sentence)
        sentence = re.sub(r"-", " - ", sentence)
        sentence = re.sub(r"\[", " [ ", sentence)
        sentence = re.sub(r"\]", " ] ", sentence)
        
        tokens = sentence.split()  
        return tokens
    
    def encode_sentence(self, sentence, lang):
        tokens = self.preprocess_sentence(sentence)
        stoi = self.src_stoi if lang == self.src_lang else self.tgt_stoi
        vec = [stoi[token] if token in stoi else self.unk_idx for token in tokens]
        vec = [self.bos_idx] + vec + [self.eos_idx]
        vec = torch.tensor(vec, dtype=torch.long)
        return vec
    
    def decode_sentence(self, tensor, lang):
        sentences_ids = tensor.tolist()
        sentences = []
        itos = self.src_itos if lang == self.src_lang else self.tgt_itos
        for idx in sentences_ids:
            words = [itos[i] for i in idx if i not in (self.bos_idx, self.eos_idx, self.pad_idx)]
            sentence = " ".join(words)
            sentences.append(sentence)
            
        return sentences