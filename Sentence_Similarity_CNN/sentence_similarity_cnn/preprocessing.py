import numpy as np
import pandas as pd
import re
from multiprocessing import Pool

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


