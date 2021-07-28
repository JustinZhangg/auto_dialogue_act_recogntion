# coding: UTF-8
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

import utils
from utils import build_dataset, build_iterator, get_time_dif


class Config(object):
    """configuration"""

    def __init__(self, dataset):
        self.model_name = 'TextRCNN'
        self.train_path = dataset + '/train.txt'  
        self.dev_path = dataset + '/dev.txt'  
        self.test_path = dataset + '/test.txt'  
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt', encoding='utf-8').readlines()]  # classes
        self.vocab_path = dataset + '/vocab.pkl'  # vocabular
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # results
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        self.dropout = 1.0  # dropout
        self.require_improvement = 1000  # stop training if no improvement after 1000
        self.num_classes = len(self.class_list)  # classes
        self.n_vocab = 0  # vocabular
        self.num_epochs = 5  # epoch
        self.batch_size = 128  # mini-batch size
        self.pad_size = 32  # length of dialogues
        self.learning_rate = 1e-3  
        self.embed = 300  # word dimensions
        self.hidden_size = 256  # lstm
        self.num_layers = 1  # lstm


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.max_pool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embedding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.max_pool(out).squeeze()
        out = self.fc(out)
        return out


class TextRCNN(object):

    def __init__(self, filename):
        # data set
        self.dataset = 'data'
        self.class_dict = {}
        self.test_class_list = []
        self.filename = filename
        self.prepare_train_valid_test_data()
        self.config = Config(self.dataset)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # ensure the result is same

        start_time = time.time()
        print("Loading data...")
        vocab, train_data, dev_data, test_data = build_dataset(self.config, True)
        self.train_iter = build_iterator(train_data, self.config)
        self.dev_iter = build_iterator(dev_data, self.config)
        self.test_iter = build_iterator(test_data, self.config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        # train
        self.config.n_vocab = len(vocab)
        self.model = Model(self.config).to(self.config.device)
        self.init_network()

    def prepare_train_valid_test_data(self):
        texts, labels = utils.load_dataset(self.filename)

        for i, l in enumerate(set(labels)):
            self.class_dict[l] = i
        self.divide_train_valid_test(texts, labels)

    def divide_train_valid_test(self, texts, labels):
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
        # split as 8:2
        with open(self.dataset + '/' + 'train.txt', 'w') as f:
            train_file = csv.writer(f, delimiter='\t')
            for i, text in enumerate(X_train):
                train_file.writerow([text, self.class_dict[y_train[i]]])
        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)

        with open(self.dataset + '/' + 'test.txt', 'w') as f:
            test_file = csv.writer(f, delimiter='\t')
            for i, text in enumerate(X_test):
                test_file.writerow([text, self.class_dict[y_test[i]]])

        self.test_class_list = [i for i in set(y_test)]

        with open(self.dataset + '/' + 'dev.txt', 'w') as f:
            valid_file = csv.writer(f, delimiter='\t')
            for i, text in enumerate(X_valid):
                valid_file.writerow([text, self.class_dict[y_valid[i]]])

    def run(self):
        print(self.model.parameters)
        self.train()

    def init_network(self, method='xavier', exclude='embedding'):
        for name, w in self.model.named_parameters():
            if exclude in name:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                continue

    def train(self):
        start_time = time.time()
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # decrease learning rate，epoch：learning rate = gamma * learning rate
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        total_batch = 0  # record the batch number
        dev_best_loss = float('inf')
        last_improve = 0  # record the validation set loss
        flag = False  # check improvement
        writer = SummaryWriter(log_dir=self.config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
        for epoch in range(self.config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, self.config.num_epochs))
            # scheduler.step() 
            for i, (trains, labels) in enumerate(self.train_iter):
                outputs = self.model(trains)
                self.model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                if total_batch % 100 == 0:
                    # the effect on training & validation set
                    true = labels.data.cpu()
                    predic = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    dev_acc, dev_loss = self.evaluate(self.config, self.model, self.dev_iter)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(self.model.state_dict(), self.config.save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                    writer.add_scalar("loss/train", loss.item(), total_batch)
                    writer.add_scalar("loss/dev", dev_loss, total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                    writer.add_scalar("acc/dev", dev_acc, total_batch)
                    self.model.train()
                total_batch += 1
                if total_batch - last_improve > self.config.require_improvement:
                    # end trainning if no improvement
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
        writer.close()
        self.test()

    def test(self):
        # test
        self.model.load_state_dict(torch.load(self.config.save_path))
        self.model.eval()
        start_time = time.time()
        test_acc, test_loss, test_report, test_confusion = self.evaluate(self.config, self.model, self.test_iter,
                                                                         test=True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

    def evaluate(self, config, model, data_iter, test=False):
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for texts, labels in data_iter:
                outputs = model(texts)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            report = metrics.classification_report(labels_all, predict_all, target_names=self.test_class_list, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(data_iter), report, confusion
        return acc, loss_total / len(data_iter)
