import json
import numpy as np
import jieba
import random
from torch.autograd import Variable
import torch

class DataLoader:

    def __init__(self, file, vocab, max_length=128):
        self.file = file
        self.vocab = vocab
        self.max_length = max_length
        self._load_data()

    def _load_data(self):
        with open(self.file, 'r') as fin:
            data = json.load(fin)
        self.total_cnt = 0
        for k in data:
            self.total_cnt += len(data[k])
        
        self.data_word = np.zeros((self.total_cnt, self.max_length),np.int32)
        self.label2scope = dict()
        i = 0
        for label in data:
            self.label2scope[label] = [i, i]
            for instance in data[label]:
                text = instance['title']
                words = jieba.lcut(text)
                cur_ref_row = self.data_word[i]
                for j, w in enumerate(words):
                    if j < self.max_length:
                        if w in self.vocab:
                            cur_ref_row[j] = self.vocab[w]
                        else:
                            cur_ref_row[j] = self.vocab['[UNK]']
                    else:
                        break
                for k in range(j, self.max_length):
                    cur_ref_row[j] = self.vocab['[PAD]']
                i += 1
            self.label2scope[label][1] = i

    def next_batch_for_one_shot(self, batch_size=20):
        s1 = []
        s2 = []
        label = []
        
        target_classes = random.sample(self.label2scope.keys(), batch_size//2)

        # 采样相似样本
        for c in target_classes:
            scope = self.label2scope[c]
            indices = np.random.choice(list(range(scope[0], scope[1])), 2)
            s1.append(self.data_word[indices[0]])
            s2.append(self.data_word[indices[1]])
            label.append(1)
        
        # 采样不相似样本
        for i in range(batch_size // 2):
            tc = random.sample(self.label2scope.keys(), 2)
            c1 = tc[0]
            c2 = tc[1]

            scope = self.label2scope[c1]
            instance1 = np.random.choice(list(range(scope[0], scope[1])), 1)
            s1.append(self.data_word[instance1[0]])

            scope = self.label2scope[c2]
            instance2 = np.random.choice(list(range(scope[0], scope[1])), 1)
            s2.append(self.data_word[instance2[0]])
            label.append(0)

        s1 = np.stack(s1, 0)
        s2 = np.stack(s2, 0)
        label = np.array(label)

        perm = np.random.permutation(len(s1))
        s1 = s1[perm]
        s2 = s2[perm]
        label = label[perm]

        return s1, s2, label

    def next_batch_for_one_shot_val(self, C, K, Q):
        support = []
        query = []
        query_label = []
        target_classes = random.sample(self.label2scope.keys(), C)

        for i, c in enumerate(target_classes):
            scope = self.label2scope[c]
            indices = np.random.choice(list(range(scope[0], scope[1])), K+Q)
            words = self.data_word[indices]
            s, q, _ = np.split(words, [K, K+Q])
            support.append(s)
            query.append(q)
            query_label += [i] * Q
        
        support = np.stack(support, 0)
        query = np.concatenate(query, 0)
        query_label = np.array(query_label)

        perm = np.random.permutation(C*Q)
        query = query[perm]
        query_label = query_label[perm]
        return support, query, query_label

    def next_one(self, C, K, Q):
        target_classes = random.sample(self.label2scope.keys(), C)

        support_set = []
        support_label = []
        query_set = []
        query_label = []

        for i, class_name in enumerate(target_classes):
            scope = self.label2scope[class_name]

            indices = np.random.choice(list(range(scope[0], scope[1])), K+Q, False)

            word = self.data_word[indices]

            support_word, query_word, _ = np.split(word, [K, K+Q])

            support_set.append(support_word)
            support_label.append([i] * K)
            query_set.append(query_word)
            query_label += [i] * Q

        support_set = np.stack(support_set, 0)
        support_label = np.stack(support_label, 0)
        query_set = np.concatenate(query_set, 0)
        query_label = np.array(query_label)

        perm = np.random.permutation(C * Q)

        query_set = query_set[perm]
        query_label = query_label[perm]

        return support_set, support_label, query_set, query_label

    def next_batch(self, B, N, K, Q):
        support_set = []
        support_label = []
        query_set = []
        label = []

        for one in range(B):
            cur_support, cur_support_label, cur_query, cur_label = self.next_one(N, K, Q)
            support_set.append(cur_support)
            support_label.append(cur_support_label)
            query_set.append(cur_query)
            label.append(cur_label)

        support = np.stack(support_set, 0)
        support_label = np.stack(support_label, 0)
        query = np.stack(query_set, 0)
        label = np.stack(label, 0)

        return support, support_label, query, label


if __name__ == "__main__":
    import utils
    vocab, embedding = utils.load_word2vec('data/tencent_embedding.txt')
    data_loader = DataLoader('data/sample_data.json', vocab)

    data_loader.next_batch(4, 5, 5, 5)