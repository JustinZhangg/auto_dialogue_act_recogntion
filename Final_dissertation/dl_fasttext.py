import random

import fasttext
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import utils


class FastText(object):
    def __init__(self, filename):
        self.filename = filename
        self.model = None
        self.process_data()
        self.train()

    def process_data(self):
        print("processing data")
        texts, labels = utils.load_dataset(self.filename)

        random.seed(0)
        random.shuffle(texts)
        random.seed(0)
        random.shuffle(labels)

        texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.05,
                                                                              stratify=labels,
                                                                              random_state=0)
        label_encoder = preprocessing.LabelEncoder()
        labels_train = label_encoder.fit_transform(labels_train)
        labels_test = label_encoder.transform(labels_test)

        with open('data/fasttext.train.txt', 'w') as f:
            for i in range(len(texts_train)):
                f.write('%s __label__%d\n' % (texts_train[i], labels_train[i]))

        with open('data/fasttext.test.txt', 'w') as f:
            for i in range(len(texts_test)):
                f.write('%s __label__%d\n' % (texts_test[i], labels_test[i]))

    def train(self):
        print("training")
        self.model = fasttext.train_supervised('data/fasttext.train.txt', epoch=10)

        print("Training Words Size : %s" % (str(len(self.model.words))))
        print("Training Labels Size :%s" % (str(len(self.model.labels))))

    def run(self):
        texts_test, labels_test = [], []
        with open('data/fasttext.test.txt', 'r') as f:
            for line in f:
                *text, label = line.strip().split(' ')
                text = ' '.join(text)
                texts_test.append(text)
                labels_test.append(label)

        label_encoder = preprocessing.LabelEncoder()
        labels_test = label_encoder.fit_transform(labels_test)
        predits = list(zip(*(self.model.predict(texts_test)[0])))[0]
        predits = label_encoder.transform(predits)

        score = metrics.f1_score(predits, labels_test, average='weighted')
        print('weighted f1-score : %.03f' % score)
