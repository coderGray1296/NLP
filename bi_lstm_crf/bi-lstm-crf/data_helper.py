import numpy as np
import pandas as pd
import sys, pickle, os, random
import json

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }

def read_corpus(corpus_path):
    """
        read corpus and return the list of samples
        :param corpus_path:
        :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
        print(lines)
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split('\t')
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data

def vocab_build(vocab_path, corpus_path, min_count):
    data = read_corpus(corpus_path)
    #print(data)
    #word2id shape is word2id[word] = [number in vacab, count in corpus]
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            # if not in word2id, then add new dict to word2id
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    # choose word that smaller than min_count
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    # delete word in low_freq_words
    for word in low_freq_words:
        del word2id[word]
    #重新写入word2id，只保留wordid
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print('字典长度为：'+ str(len(word2id)))
    with open(vocab_path, 'w') as fw:
        json.dump(word2id, fw, ensure_ascii=False)

# turn sentence(list) to id(list) seeking from word2id(dict)
def sentence2id(sent, word2id):
    """
        :param sent:
        :param word2id:
        :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

def read_dictionary(vocab_path):
    """
        :param vocab_path:
        :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'r') as fr:
        word2id = json.load(fr)
    print('vocab_size:', len(word2id))
    return word2id

def random_embedding(vocab, embedding_dim):
    """
        :param vocab:
        :param embedding_dim:
        :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat
#给seq进行0填充，返回填充后的seqs和原seqs的长度list
def pad_sequences(sequences, pad_mark=0):
    """
        :param sequences:
        :param pad_mark:
        :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """
        :param data:
        :param batch_size:
        :param vocab:
        :param tag2label:
        :param shuffle:
        :return:
    """
    if shuffle:
        random.shuffle(data)
    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)
    if len(seqs) != 0:
        yield seqs, labels


#vocab_build('./data/vocab.json','./data/test.txt',0)
word2id = read_dictionary('./data/vocab.json')
word_embedding = random_embedding(word2id, 10)
print(word_embedding)