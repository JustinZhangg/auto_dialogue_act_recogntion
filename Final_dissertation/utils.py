import csv
import os
import pickle as pkl
import time
from datetime import timedelta
from typing import Tuple

import torch
from tqdm import tqdm


def load_dataset(filename) -> Tuple:
    labels_count = {}
    texts, labels = [], []
    with open(filename, 'r') as f:
        for i, line in enumerate(csv.reader(f)):
            if i == 0:
                continue
            label = line[2]
            text = line[5].replace('/', " ")
            if not text or not label:
                continue
            if label not in labels_count:
                labels_count[label] = [text]
            else:
                labels_count[label].append(text)

    for lab in labels_count:
        text_list = labels_count[lab]
        text_len = len(text_list)
        if text_len == 1:
            continue
        labels.extend([lab] * text_len)
        texts.extend(text_list)

    with open('./data/class.txt', 'w') as f:
        for c in set(labels):
            f.write(c)
            f.write('\n')
    return texts, labels


MAX_VOCAB_SIZE = 10000  # length limit
UNK, PAD = '<UNK>', '<PAD>'  # unknown words，padding tokens


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def word_tokenizer(x):
    return x.split(' ')


def char_tokenizer(x):
    return [y for y in x]


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = word_tokenizer  # split by ' '，word-level
    else:
        tokenizer = char_tokenizer  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dl_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                line_split = lin.split('\t')
                if len(line_split) < 2:
                    print(lin)
                content, label = line_split
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dl_dataset(config.train_path, config.pad_size)
    dev = load_dl_dataset(config.dev_path, config.pad_size)
    test = load_dl_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterator(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # check whether batch number is int
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, data):
        x = torch.LongTensor([_[0] for _ in data]).to(self.device)
        y = torch.LongTensor([_[1] for _ in data]).to(self.device)

        # length before padding(more than pad_size then set to pad_size)
        seq_len = torch.LongTensor([_[2] for _ in data]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """get time used"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
