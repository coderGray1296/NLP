# encoding=utf-8
import nltk
from nltk import ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
import util
import numpy as np
import pickle as pkl
from gensim.models.keyedvectors import KeyedVectors
from functools import reduce

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

seq_label_dict = dict([('I-ORGANIZATION', 4),
                       ('B-ORGANIZATION', 3),
                       ('I-TELEPHONE', 0),
                       ('B-TELEPHONE', 0),
                       ('I-LOCATION', 0),
                       ('B-LOCATION', 0),
                       ('I-PERCENT', 0),
                       ('B-PERCENT', 0),
                       ('I-PERSON', 2),
                       ('B-PERSON', 1),
                       ('I-EMAIL', 0),
                       ('B-EMAIL', 0),
                       ('I-MONEY', 0),
                       ('B-MONEY', 0),
                       ('I-PLACE', 0),
                       ('B-PLACE', 0),
                       ('I-DATE', 0),
                       ('B-DATE', 0),
                       ('I-TIME', 0),
                       ('B-TIME', 0),
                       ('I-URL', 0),
                       ('B-URL', 0),
                       ('O', 0)]
                      )
reverse_seq_label_dict = {0: 'O',
                          1: 'B-PERSON',
                          2: 'I-PERSON',
                          3: 'B-ORGANIZATION',
                          4: 'I-ORGANIZATION'}
label_map = reverse_seq_label_dict
sentiment_dict = {'_': 0,
                  'positive': 1,
                  'negative': 2,
                  'neutral': 3}
sentiment_map = {'_': 0,
                 'B-positive': 1,
                 'I-positive': 2,
                 'B-negative': 3,
                 'I-negative': 4,
                 'B-neutral': 5,
                 'I-neutral': 6}

senti_label_map = {}
for k in sentiment_map:
    senti_label_map[sentiment_map[k]] = k
BI_dict = {}
senti_BI_dict = {1: 2,
                 3: 4,
                 5: 6}

#类似于senti_BI_dict
def match(d):
    for k in d:
        begin = k.split('-')
        if len(begin) > 1 and begin[0] == 'B':
            e = 'I-' + begin[-1]
            if e in d:
                BI_dict[seq_label_dict[k]] = seq_label_dict[e]
            else:
                BI_dict[seq_label_dict[k]] = -1


match(seq_label_dict)


def read_file(name):
    f = open(name)
    data = []
    twitter = ''
    seq_label = ''
    sentiment = ''
    for line in f:
        if '## Tweet' in line:
            continue
        line = line.strip().split()
        if len(line) == 0:
            idx_seq_label = []
            for tmp in seq_label.split():
                idx_seq_label.append(seq_label_dict[tmp])
            tmp_sentiment_list = []
            for s in sentiment.split():
                if s not in sentiment_dict:
                    sentiment_dict[s] = len(sentiment_dict)
                tmp_sentiment_list.append(sentiment_dict[s])
            sentiment_list = []
            index = 0
            length = len(tmp_sentiment_list)
            while index < length:
                if idx_seq_label[index] == seq_label_dict['O']:
                    sentiment_list.append(sentiment_dict['_'])
                    index += 1
                else:
                    begin_label = reverse_seq_label_dict[idx_seq_label[index]]
                    # print begin_label
                    # print(begin_label)
                    begin_label = begin_label.split('-')[1]
                    tmp_index = index + 1
                    # while tmp_index < length and idx_seq_label[tmp_index] != seq_label_dict['O']:
                    while tmp_index < length and idx_seq_label[tmp_index] == senti_BI_dict[idx_seq_label[index]]:
                        # print reverse_seq_label_dict[idx_seq_label[tmp_index]].split('-')
                        next_label = reverse_seq_label_dict[idx_seq_label[tmp_index]].split('-')[1]
                        if next_label == begin_label:
                            tmp_index += 1
                        else:
                            break
                    if tmp_sentiment_list[index] == sentiment_dict['_']:
                        senti = [tmp_sentiment_list[index]] * (tmp_index - index)
                    elif tmp_sentiment_list[index] == sentiment_dict['positive']:
                        senti = [sentiment_map['B-positive']] + [sentiment_map['I-positive']] * (tmp_index - index - 1)
                    elif tmp_sentiment_list[index] == sentiment_dict['negative']:
                        senti = [sentiment_map['B-negative']] + [sentiment_map['I-negative']] * (tmp_index - index - 1)
                    elif tmp_sentiment_list[index] == sentiment_dict['neutral']:
                        senti = [sentiment_map['B-neutral']] + [sentiment_map['I-neutral']] * (tmp_index - index - 1)
                    else:
                        print('No match!!!')
                        exit(0)
                    sentiment_list = sentiment_list + senti
                    index = tmp_index
            pos = [x[1] for x in nltk.pos_tag(twitter.split())]
            # print(twitter)
            # print(pos)
            # print(idx_seq_label)
            # print(sentiment_list)
            # print()
            pos = reduce(lambda x1, x2: x1 + ' ' + x2, pos)
            char_list = twitter.split()
            data.append((twitter.lower(), idx_seq_label, sentiment_list, pos, char_list))
            # print twitter
            # print seq_label
            # print sentiment
            # print
            twitter = ''
            seq_label = ''
            sentiment = ''
            continue
        if '@' in line[0]:
            twitter += 'USRTOK '
        elif 'http:' in line[0]:
            twitter += 'URLTOK '
        else:
            twitter += line[0] + ' '
        seq_label += line[1] + ' '
        sentiment += line[2] + ' '
    f.close()
    print(sentiment_dict)
    return data


def get_term3(text, label, sentiment, BI_dict=BI_dict):
    length = len(text)
    assert length == len(label)

    result = {}
    index = 0
    while index < length:
        if label[index] != 0 and (label[index] in BI_dict):
            k = index + 1
            while k < length and (label[k] == BI_dict[label[index]]):
                k += 1
            tmp = text[index:k]
            term = ''
            for t in tmp:
                if type(t) != str:
                    t = str(t)
                term += t + ' '
            term = term.strip()
            flag = True
            for lll in sentiment[index:k]:
                if lll == 0:
                    flag = False
            if flag:
                if term not in result:
                    result[term] = set()
                result[term].add(((index, k - 1), tuple(label[index:k]), tuple(sentiment[index:k])))
            index = k
        else:
            index += 1;
    return result


def get_term2(text, label, sentiment, BI_dict=BI_dict):
    length = len(text)
    assert length == len(label)

    result = {}
    index = 0
    while index < length:
        if label[index] != 0 and (label[index] in BI_dict):
            k = index + 1
            while k < length and (label[k] == BI_dict[label[index]]):
                k += 1
            tmp = text[index:k]
            term = ''
            for t in tmp:
                if type(t) != str:
                    t = str(t)
                term += t + ' '
            term = term.strip()
            if term not in result:
                result[term] = set()
            result[term].add(((index, k - 1), tuple(label[index:k]), tuple(sentiment[index:k])))
            index = k
        else:
            index += 1;
    return result


def get_term(text, label, tag_num=3, BI_dict=BI_dict):
    length = len(text)
    assert length == len(label)

    result = {}
    index = 0
    while index < length:
        if tag_num == 3:
            if label[index] != 0 and (label[index] in BI_dict):
                k = index + 1
                while k < length and (label[k] == BI_dict[label[index]]):
                    k += 1
                tmp = text[index:k]
                term = ''
                for t in tmp:
                    if type(t) != str:
                        t = str(t)
                    term += t + ' '
                term = term.strip()
                if term not in result:
                    result[term] = set()
                # result[term].add((index, k-1))
                result[term].add(((index, k - 1), tuple(label[index:k])))
                index = k
            else:
                index += 1
        else:
            raise ValueError("Label Size Error!!!")
    return result


def evaluation(pred, true):
    pred_num = 0
    true_num = 0
    pred_correct = 0
    for p, t in zip(pred, true):
        for k in p:
            pred_num += len(p[k])
        for k in t:
            true_num += len(t[k])
        for k in p:
            if k in t:
                for index in p[k]:
                    if index in t[k]:
                        pred_correct += 1
    print('predict num: {}'.format(pred_num))
    print('ground  num: {}'.format(true_num))
    print('correct num: {}'.format(pred_correct))
    p = 0.0
    if pred_num:
        p = pred_correct * 1. / pred_num
    r = 0.0
    if true_num:
        r = pred_correct * 1. / true_num

    f = 0.0
    if (p + r) != 0:
        f = 2 * p * r / (p + r)

    return p, r, f


class Data():

    def __init__(self,
                 dataset,
                 test_file_id=1,  # from 1 to 10
                 path='./MitchellEtAI/',
                 max_len=41,
                 batch_size=32,
                 max_char_len=10,
                 max_target_len=3,
                 window_size=5):
        # data = [read_file(path + dataset + '/10-fold/test.' + str(i+1)) for i in range(10)]
        # test_data = data[test_file_id]
        # train_data= reduce(lambda x1,x2:x1+x2, data[:test_file_id] + data[test_file_id+1:])
        train_data = read_file(path + dataset + '/10-fold/train.' + str(test_file_id))
        test_data = read_file(path + dataset + '/10-fold/test.' + str(test_file_id))
        test_text = [x[0] for x in test_data]
        test_seq_label = [x[1] for x in test_data]
        test_seq_senti = [x[2] for x in test_data]
        test_pos = [x[3] for x in test_data]
        test_char = [x[4] for x in test_data]
        self.test_raw_text = test_text

        train_text = [x[0] for x in train_data]
        train_seq_label = [x[1] for x in train_data]
        train_seq_senti = [x[2] for x in train_data]
        train_pos = [x[3] for x in train_data]
        train_char = [x[4] for x in train_data]

        tmp_train_char = reduce(lambda x1, x2: x1 + x2, [x for x in train_char])

        self.char_dict = util.get_char_dict(tmp_train_char)
        self.word_dict = util.get_dict(train_text)
        self.pos_dict = util.get_dict(train_pos)
        self.seq_label_dict = 5  # 0~4 for O, B-person, I-Person, B-organization, I-organization
        self.senti_label_size = 7  # 0~6 for None, B_Negative,I_Negative, B_Neutral, I_Neutral B_Positive , I_Positive
        print(' seq label size: {}'.format(self.seq_label_dict))

        word2vec_path = '~/glove/GoogleNews-vectors-negative300.bin'
        binary = True;
        emb_dim = 300
        if dataset == 'es':
            word2vec_path = '~/glove/SBW-vectors-300-min5.bin';
            emb_dim = 300  # 300 dim
            # word2vec_path = '~/glove/fasttext-sbwc.3.6.e20-es.bin'
            # word2vec_path = '~/glove/glove-sbwc.i25-es.bin'
            word2vec_path = '/home/mdh/glove/embedding_file';
            binary = False;
            emb_dim = 200  # 200 dim

        word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=binary)
        self.embedding = np.random.uniform(-0.1, 0.1, (len(self.word_dict), emb_dim)).astype('float32')
        for k in self.word_dict:
            if k in word_vectors:
                self.embedding[self.word_dict[k]][:] = word_vectors[k]
        dump_path = ''
        if dataset == 'en':
            dump_path = './embeddings/acl2015-en.pkl'
        else:
            dump_path = './embeddings/acl2015-fast-es.pkl'
        with open(dump_path, 'wb') as f:
            pkl.dump(self.embedding, f)
        dump_path = ''
        '''
        if dataset == 'en':
            dump_path = './embeddings/acl2015-en.pkl'
        else:
            dump_path = './embeddings/acl2015-es.pkl'
        with open(dump_path, 'rb') as f:
            self.embedding = pkl.load(f)
        '''

        self.char_size = len(self.char_dict)
        self.vocab_size = len(self.word_dict)
        self.label_size = 3  # no use
        self.pos_size = len(self.pos_dict)
        self.ner_size = 10  # no use
        self.max_len = max_len
        self.batch_size = batch_size
        self.term_label_size = len(reverse_seq_label_dict)
        self.window_size = window_size

        train_char = util.char2idx(train_char, self.char_dict)
        test_char = util.char2idx(test_char, self.char_dict)

        train_text = util.word2idx(train_text, self.word_dict)
        test_text = util.word2idx(test_text, self.word_dict)

        train_text, train_seq_len, train_mask = util.padding(train_text, max_len)
        test_text, test_seq_len, test_mask = util.padding(test_text, max_len)
        train_context = util.context_window(np.asarray(train_text), self.window_size)
        test_context = util.context_window(np.asarray(test_text), self.window_size)

        train_char, train_char_len = util.char_padding(train_char, max_len, max_char_len)
        test_char, test_char_len = util.char_padding(test_char, max_len, max_char_len)

        train_seq_label, _, _ = util.padding(train_seq_label, max_len)
        test_seq_label, _, _ = util.padding(test_seq_label, max_len)

        train_seq_senti, _, _ = util.padding(train_seq_senti, max_len)
        test_seq_senti, _, _ = util.padding(test_seq_senti, max_len)

        train_pos = util.word2idx(train_pos, self.pos_dict)
        test_pos = util.word2idx(test_pos, self.pos_dict)
        train_pos, _, _ = util.padding(train_pos, max_len)
        test_pos, _, _ = util.padding(test_pos, max_len)

        self.train_pos = np.asarray(train_pos[:]).astype('int32')
        self.train_text = np.asarray(train_text[:]).astype('int32')
        self.train_mask = np.asarray(train_mask[:]).astype('float32')
        self.train_char = np.asarray(train_char[:]).astype('int32')
        self.train_context = np.asarray(train_context[:]).astype('int32')
        self.train_seq_len = np.asarray(train_seq_len[:]).astype('int32')
        self.train_char_len = np.asarray(train_char_len[:]).astype('int32')
        self.train_seq_label = np.asarray(train_seq_label[:]).astype('int32')
        self.train_seq_senti = np.asarray(train_seq_senti[:]).astype('int32')

        print('Train pos  shape     : {}'.format(self.train_pos.shape))
        print('Train text shape     : {}'.format(self.train_text.shape))
        print('Train mask shape     : {}'.format(self.train_mask.shape))
        print('Train char shape     : {}'.format(self.train_char.shape))
        print('Train context shape  : {}'.format(self.train_context.shape))
        print('Train text len shape : {}'.format(self.train_seq_len.shape))
        print('Train char len shape : {}'.format(self.train_char_len.shape))
        print('Train seq label shape: {}'.format(self.train_seq_label.shape))
        print('Train seq senti shape: {}\n'.format(self.train_seq_senti.shape))

        self.test_pos = np.asarray(test_pos).astype('int32')
        self.test_text = np.asarray(test_text).astype('int32')
        self.test_mask = np.asarray(test_mask).astype('float32')
        self.test_char = np.asarray(test_char).astype('int32')
        self.test_context = np.asarray(test_context).astype('int32')
        self.test_seq_len = np.asarray(test_seq_len).astype('int32')
        self.test_char_len = np.asarray(test_char_len).astype('int32')
        self.test_seq_label = np.asarray(test_seq_label).astype('int32')
        self.test_seq_senti = np.asarray(test_seq_senti).astype('int32')

        print('Test pos shape       : {}'.format(self.test_pos.shape))
        print('Test text shape      : {}'.format(self.test_text.shape))
        print('Test mask shape      : {}'.format(self.test_mask.shape))
        print('Test char shape      : {}'.format(self.test_char.shape))
        print('Test context shape   : {}'.format(self.test_context.shape))
        print('Test text len shape  : {}'.format(self.test_seq_len.shape))
        print('Test char len shape  : {}'.format(self.test_char_len.shape))
        print('Test seq label shape : {}'.format(self.test_seq_label.shape))
        print('Test seq senti shape : {}\n'.format(self.test_seq_senti.shape))

        self.train_epoch = self.train_text.shape[0] // self.batch_size
        if self.train_text.shape[0] % self.batch_size:
            self.train_epoch += 1

        self.test_epoch = self.test_text.shape[0] // self.batch_size
        if self.test_text.shape[0] % self.batch_size:
            self.test_epoch += 1

        print('Train epoch: {}'.format(self.train_epoch))
        # print 'Valid epoch: {}'.format(self.valid_epoch)
        print('Test  epoch: {}'.format(self.test_epoch))

        self.shuffle()
        match(seq_label_dict)
        s = sorted(seq_label_dict.items(), key=lambda x: x[1], reverse=True)
        for x in s:
            print(x)

        for k in BI_dict:
            print(k, ' : ', BI_dict[k])

        self.text_dict_reverse = {}
        for k in self.word_dict:
            self.text_dict_reverse[self.word_dict[k]] = k

    def reverse(self, data, d):
        return [d[d_] for d_ in data]

    def shuffle(self):
        tmp = list(zip(self.train_pos, self.train_text, self.train_char, self.train_seq_len, self.train_char_len,
                       self.train_seq_label, self.train_seq_senti, self.train_mask))
        np.random.shuffle(tmp)
        a, b, c, f, g, h, i, j = [], [], [], [], [], [], [], []
        for a_, b_, c_, f_, g_, h_, i_, j_ in tmp:
            a.append(a_)
            b.append(b_)
            c.append(c_)
            f.append(f_)
            g.append(g_)
            h.append(h_)
            i.append(i_)
            j.append(j_)

        self.train_pos = np.asarray(a)
        self.train_text = np.asarray(b)
        self.train_char = np.asarray(c)
        self.train_seq_len = np.asarray(f)
        self.train_char_len = np.asarray(g)
        self.train_seq_label = np.asarray(h)
        self.train_seq_senti = np.asarray(i)
        self.train_mask = np.asarray(j)


if __name__ == '__main__':
    d = Data('es')
