import numpy as np
import pandas as pd
import re
from multiprocessing import Pool
import word2vec

class dataset(object):
    def __init__(self, s1, s2, label):
        self.s1 = s1
        self.s2 = s2
        self.label = label
        self.index_in_epoch = 0
        self.examples_nums = len(label)
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.examples_nums:
            #reset the start and end
            start = 0
            self.index_in_epoch = batch_size
            #finished one epoch
            self.epochs_completed += 1
            #shuffle the data
            perm = np.arange(self.examples_nums)
            np.random.shuffle(perm)
            self.s1 = self.s1[perm]
            self.s2 = self.s2[perm]
            self.label = self.label[perm]
            #error prediction
            assert batch_size <= self.examples_nums
        end = self.index_in_epoch
        return np.array(np.array(self.s1[start:end]), np.array(self.s2[start:end]), np.array(self.label[start:end]))


def clean_str(string):
    #
    #对句子相似度任务进行字符清洗
    #
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def padding_sentence(s1, s2):
    s1_length_max = max([len(s) for s in s1])
    s2_length_max = max([len(s) for s in s2])
    sentence_length_max = max(s1_length_max, s2_length_max)
    sentence_num = s1.shape[0]
    s1_padding = np.zeros([sentence_num, sentence_length_max], dtype=int)
    s2_padding = np.zeros([sentence_num, sentence_length_max], dtype=int)

    for i, s in enumerate(s1):
        s1_padding[i][:len(s)] = s

    for i, s in enumerate(s2):
        s2_padding[i][:len(s)] = s

    print("FINISHED EMBADDING!")
    return s1_padding, s2_padding



def read_data_sets(train_dir = 'example.txt'):
    #
    # s1代表数据集的句子1
    # s2代表数据集的句子2
    # score代表相似度
    # sample_num代表数据总共有多少行
    #

    df_sick = pd.read_csv(train_dir, sep="\t", usecols=[1, 2, 4], names=['s1', 's2', 'score'],
                          dtype={'s1': object, 's2': object, 'score': object})
    #delete the first row
    df_sick = df_sick.drop([0])

    s1 = df_sick.s1.values
    s2 = df_sick.s2.values
    score = [float(s) for s in df_sick.score.values]
    score = np.asarray(score, dtype=np.float32)
    sample_num = len(score)

    global sr_word2id, word_embadding
    sr_word2id, word_embadding = build_glove_dic()

    #word to id
    s1 = np.array(word2id(s1))
    s2 = np.array(word2id(s2))

    #padding and shuffle
    s1, s2 = padding_sentence(s1, s2)
    index = np.random.permutation(sample_num)
    s1 = s1[index]
    s2 = s2[index]
    score = score[index]

    return s1, s2, score

def word2id(s):
    result = []
    for ss in s:
        temp = []
        ss = clean_str(ss)
        ss_split = ss.split(' ')
        for ss_split_s in ss_split:
            if ss_split_s in sr_word2id:
                temp.append(sr_word2id[ss_split_s])
            else:
                temp.append(0)
        result.append(temp)
    return result

def build_glove_dic():
    glove_path = 'glove.6B.50d.txt'
    wv = word2vec.load(glove_path)
    #获取词表
    vocab = wv.vocab
    #根据词表建立索引
    sr_word2id = pd.Series(range(1, len(vocab)+1), index=vocab)
    sr_word2id['<unk>'] = 0
    #获取wr对象的所有向量
    word_embadding = wv.vectors
    #计算每一列的均值
    word_mean = np.mean(word_embadding, axis=0)
    #将均值向量加在word_embedding上，维度为[40000+1,50]
    word_embadding = np.vstack([word_mean, word_embadding])
    return sr_word2id, word_embadding

def batch_iter(data, batch_size, num_epochs, shuffled=True):
    data = np.array(data)
    data_length = len(data)
    num_batchs_per_epoch = int((data_length-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffled:
            index = np.random.permutation(np.arange(data_length))
            data_shuffled = data[index]
        else:
            data_shuffled = data
        for num_batch in range(num_batchs_per_epoch):
            start = num_batch * batch_size
            end = min((num_batch + 1)*batch_size, data_length)
            yield data_shuffled[start:end]


#s1, s2, score = read_data_sets('example.txt')

