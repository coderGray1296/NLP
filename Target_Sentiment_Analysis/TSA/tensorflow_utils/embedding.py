# encoding=utf-8
import numpy as np

np.random.seed(123)


class Embedding(object):

    def __init__(self, vocab_dict, embedding_size, path='/home/mdh/glove/glove.840B.300d.txt'):

        wed = {}
        f = open(path)
        for line in f:
            line = line.strip().split()
            wed[line[0]] = [float(x) for x in line[1:]]
        f.close()

        self.embedding = np.random.uniform(-0.1, 0.1, size=(len(vocab_dict), embedding_size)).astype('float32')
        for w in vocab_dict:
            if w in wed:
                word = vocab_dict[w]
                embe = wed[w]
                self.embedding[word, :] = embe
