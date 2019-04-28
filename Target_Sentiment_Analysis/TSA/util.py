# encoding=utf-8
import numpy as np


def get_dict(text):
    word_dict = {}
    word_dict['PADDING'] = 0
    word_dict['UNKNOWN'] = 1

    for t in text:
        t = t.split()
        for w in t:
            if w not in word_dict:
                word_dict[w] = len(word_dict)
    return word_dict


def get_char_dict(text):
    word_dict = {}
    word_dict['PADDING'] = 0
    word_dict['UNKNOWN'] = 1

    for t in text:
        for c in t:
            if c not in word_dict:
                word_dict[c] = len(word_dict)
    return word_dict


def word2idx(data, word_dict):
    result = []
    for d in data:
        idx_list = []
        for w in d.split():
            if w in word_dict:
                idx_list.append(word_dict[w])
            else:
                word_dict[w] = len(word_dict)
                idx_list.append(word_dict[w])
            # idx_list.append(word_dict['UNKNOWN'])
        result.append(idx_list)
    return result


def char2idx(data, char_dict):
    result = []
    for sent in data:
        idx_list = []
        for word in sent:
            char_idx = []
            for char in word.strip():
                if char in char_dict:
                    char_idx.append(char_dict[char])
                else:
                    char_idx.append(char_dict['UNKNOWN'])
            idx_list.append(char_idx)
        result.append(idx_list)
    return result


def padding(data, max_len):
    result = []
    seq_len = []
    mask = []
    for d in data:
        d = d[:max_len]
        seq_len.append(len(d))
        result.append(d + [0] * (max_len - len(d)))
        mask.append([1.0] * len(d) + [0.0] * (max_len - len(d)))
    return result, seq_len, mask


def char_padding(data, max_len, max_char_len):
    padded = []
    seq_len = []
    for sent in data:
        sent = sent[:max_len]
        char_len = [len(x[:max_char_len]) for x in sent] + [0] * (max_len - len(sent))
        sent = sent + [[0]] * (max_len - len(sent))
        pad_sent = []
        for word in sent:
            pad_sent.append(word[:max_char_len] + [0] * (max_char_len - len(word)))
        padded.append(pad_sent)
        seq_len.append(char_len)
    return padded, seq_len


def context_window(text, window_size):
    context = []
    text_shape = text.shape
    left = np.zeros((text_shape[0], window_size // 2))
    right = np.zeros((text_shape[0], window_size // 2))
    tmp = np.concatenate((left, text, right), axis=1)
    for i in range(text_shape[1]):
        context.append(tmp[:, i:i + window_size])
    context = np.asarray(context).swapaxes(0, 1)
    return context
