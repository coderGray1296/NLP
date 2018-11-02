import numpy as np
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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

#return sentences and label of array
def load_data(positive_data_file, negative_data_file):

    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    text = positive_examples + negative_examples
    text = [clean_str(t) for t in text]

    #generate labels
    positive_examples = [[0, 1] for _ in positive_examples]
    negative_examples = [[1, 0] for _ in negative_examples]
    #print(positive_examples)
    y = np.concatenate([positive_examples + negative_examples], 0)
    return text, y
    #print(text, y)

#generator batch_data
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batchs_per_epoch = int((data_size-1)/batch_size) + 1

    for epoch in range(num_epochs):
        #shuffle data ever epoch
        if shuffle:
            shuffled_data = np.random.permutation(data)
        else:
            shuffled_data = data
        for num_batch in range(num_batchs_per_epoch):
            start_index = num_batch * batch_size
            end_index = min((num_batch + 1) * batch_size, data_size)
            yield shuffled_data[start_index: end_index]


data = load_data('./data/positive.txt', './data/negative.txt')




