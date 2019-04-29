# encoding=utf-8
seed = 2
import random

random.seed(seed)
import numpy as np

np.random.seed(seed)
import tensorflow as tf

tf.set_random_seed(seed)

import argparse
from acl2015_data import Data, get_term, get_term2, get_term3, evaluation, label_map, senti_label_map, BI_dict, \
    senti_BI_dict

from tagger import Model
import sys

if __name__ == '__main__':

    data = 'es'  # dataset: en: english es: spanish
    test_file_id = 10
    d = Data(data, test_file_id=test_file_id)

    word_embedding_size = 300
    if data == 'es':
        word_embedding_size = 200
    char_embedding_size = 50
    pos_embedding_size = 50
    ner_embedding_size = 50

    parse = argparse.ArgumentParser()
    parse.add_argument('--cell', type=str, default='gru')
    parse.add_argument('--word_embedding_size', type=int, default=word_embedding_size)
    parse.add_argument('--char_embedding_size', type=int, default=char_embedding_size)
    parse.add_argument('--pos_embedding_size', type=int, default=pos_embedding_size)
    parse.add_argument('--ner_embedding_size', type=int, default=ner_embedding_size)
    parse.add_argument('--max_character_len', type=int, default=10)
    parse.add_argument('--char_hidden_size', type=int, default=char_embedding_size)
    parse.add_argument('--senti_label_size', type=int, default=d.senti_label_size)
    parse.add_argument('--valid_frequency', type=int, default=2)
    parse.add_argument('--max_grad_norm', type=float, default=20.0)
    parse.add_argument('--learning_rate', type=type(0.5), default=1e-3)
    parse.add_argument('--dropout_rate', type=type(0.5), default=0.5)
    parse.add_argument('--hidden_size', type=int, default=word_embedding_size)
    parse.add_argument('--vocab_size', type=int, default=d.vocab_size)
    parse.add_argument('--label_size', type=int, default=d.term_label_size)
    parse.add_argument('--char_size', type=int, default=d.char_size)
    parse.add_argument('--pos_size', type=int, default=d.pos_size)
    parse.add_argument('--ner_size', type=int, default=d.ner_size)
    parse.add_argument('--max_len', type=int, default=d.max_len)
    parse.add_argument('--dropout', type=type(False), default=True)
    parse.add_argument('--use_crf', type=type(False), default=True)
    parse.add_argument('--use_cnn', type=type(False), default=False)
    parse.add_argument('--use_reg', type=type(False), default=False)
    parse.add_argument('--bi_rnn', type=type(False), default=True)
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('--embedding', type=type(d.embedding), default=d.embedding)
    # parse.add_argument('--embedding' , type=type(None), default=None)
    parse.add_argument('--batch_size', type=int, default=d.batch_size)
    parse.add_argument('--cnn_kernels', type=int, default=300)
    parse.add_argument('--window_size', type=int, default=d.window_size)
    parse.add_argument('--rnn_layer_num', type=int, default=1)
    parse.add_argument('--char_embedding', type=type(None), default=None)
    parse.add_argument('--l2_coefficient', type=float, default=0.0)

    args = parse.parse_args()
    print(args)
    model = Model(args)

    max_test_f = 0.0
    max_senti_test_f = 0.0
    learning_rate = 1e-3
    test_p, test_r, test_f = 0.0, 0.0, 0.0
    test_result = []
    senti_test_result = []
    # config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epochs):
            print('Epoch: {}, Epoch num: {}'.format(epoch, d.train_epoch))
            for i in range(d.train_epoch):
                train_fd = {
                    model.text: d.train_text[i * args.batch_size:(i + 1) * args.batch_size],
                    model.char: d.train_char[i * args.batch_size:(i + 1) * args.batch_size],
                    model.mask: d.train_mask[i * args.batch_size:(i + 1) * args.batch_size],
                    model.pos: d.train_pos[i * args.batch_size:(i + 1) * args.batch_size],
                    model.ner: d.train_pos[i * args.batch_size:(i + 1) * args.batch_size],
                    model.seq_len: d.train_seq_len[i * args.batch_size:(i + 1) * args.batch_size],
                    model.char_len: d.train_char_len[i * args.batch_size:(i + 1) * args.batch_size],
                    model.label: d.train_seq_label[i * args.batch_size:(i + 1) * args.batch_size],
                    model.senti_label: d.train_seq_senti[i * args.batch_size:(i + 1) * args.batch_size],
                    model.context: d.train_context[i * args.batch_size:(i + 1) * args.batch_size],
                    model.learning_rate: learning_rate,
                    model.dropout_rate: 0.5
                }
                loss, _ = sess.run([model.loss, model.train_op], feed_dict=train_fd)
                if True:
                    print('Epoch: {}, batch: {}, batch loss: {}'.format(epoch, i, loss))
            if True:
                test_y_pred, test_y_true, test_senti_y_pred, test_senti_y_true, test_joint_senti_y_pred, test_joint_senti_y_true = [], [], [], [], [], []
                test_joint_senti_y_pred1, test_joint_senti_y_true1 = [], []
                ori_y_pred, ori_y_true, ori_text, ori_senti_y_pred, ori_senti_y_true = [], [], [], [], []
                for j in range(d.test_epoch):
                    test_fd = {
                        model.text: d.test_text[j * args.batch_size:(j + 1) * args.batch_size],
                        model.char: d.test_char[j * args.batch_size:(j + 1) * args.batch_size],
                        model.mask: d.test_mask[j * args.batch_size:(j + 1) * args.batch_size],
                        model.pos: d.test_pos[j * args.batch_size:(j + 1) * args.batch_size],
                        model.ner: d.test_pos[j * args.batch_size:(j + 1) * args.batch_size],
                        model.seq_len: d.test_seq_len[j * args.batch_size:(j + 1) * args.batch_size],
                        model.char_len: d.test_char_len[j * args.batch_size:(j + 1) * args.batch_size],
                        model.label: d.test_seq_label[j * args.batch_size:(j + 1) * args.batch_size],
                        model.senti_label: d.test_seq_senti[j * args.batch_size:(j + 1) * args.batch_size],
                        model.context: d.test_context[j * args.batch_size:(j + 1) * args.batch_size],
                        model.dropout_rate: 1.0
                    }
                    if args.use_crf:
                        batch_y_pred = model.predict_batch(sess, test_fd)
                        batch_senti_y_pred = model.predict_batch_senti(sess, test_fd)
                    else:
                        batch_y_pred, batch_senti_y_pred = sess.run([model.y_pred, model.senti_y_pred],
                                                                    feed_dict=test_fd)
                    batch_y_pred = list(batch_y_pred)
                    batch_senti_y_pred = list(batch_senti_y_pred)
                    batch_y_true = list(d.test_seq_label[j * args.batch_size:(j + 1) * args.batch_size])
                    batch_senti_y_true = list(d.test_seq_senti[j * args.batch_size:(j + 1) * args.batch_size])
                    batch_seq_len = list(d.test_seq_len[j * args.batch_size:(j + 1) * args.batch_size])
                    batch_text = [list(x) for x in list(d.test_text[j * args.batch_size:(j + 1) * args.batch_size])]
                    for p, t, te, s, sp, st in zip(batch_y_pred, batch_y_true, batch_text, batch_seq_len,
                                                   batch_senti_y_pred, batch_senti_y_true):
                        p_ = list(p)[:s]
                        t_ = list(t)[:s]
                        te_ = list(te)[:s]
                        sp_ = list(sp)[:s]
                        st_ = list(st)[:s]
                        ori_y_pred.append(p_)
                        ori_y_true.append(t_)
                        ori_text.append(te_)
                        ori_senti_y_pred.append(sp_)
                        ori_senti_y_true.append(st_)

                        test_y_pred.append(get_term(te_, p_, BI_dict=BI_dict))
                        test_y_true.append(get_term(te_, t_, BI_dict=BI_dict))
                        test_senti_y_pred.append(get_term(te_, sp_, BI_dict=senti_BI_dict))
                        test_senti_y_true.append(get_term(te_, st_, BI_dict=senti_BI_dict))
                        test_joint_senti_y_pred.append(get_term2(te_, p_, sp_, BI_dict=BI_dict))
                        # test_joint_senti_y_true.append(get_term2(te_,p_,st_,BI_dict=BI_dict))
                        test_joint_senti_y_true.append(get_term2(te_, t_, st_, BI_dict=BI_dict))
                        # contraint joint # our results use get_term3 rather than get_term2
                        test_joint_senti_y_pred1.append(get_term3(te_, p_, sp_, BI_dict=BI_dict))
                        test_joint_senti_y_true1.append(get_term3(te_, t_, st_, BI_dict=BI_dict))
                test_p, test_r, test_f = evaluation(test_y_pred, test_y_true)

                test_senti_p, test_senti_r, test_senti_f = evaluation(test_senti_y_pred, test_senti_y_true)
                test_joint_senti_p, test_joint_senti_r, test_joint_senti_f = evaluation(test_joint_senti_y_pred,
                                                                                        test_joint_senti_y_true)
                test_joint_senti_p1, test_joint_senti_r1, test_joint_senti_f1 = evaluation(test_joint_senti_y_pred1,
                                                                                           test_joint_senti_y_true1)

                sa_p = test_joint_senti_p
                sa_r = test_senti_r
                sa_f = 0.0
                if sa_p + sa_r:
                    sa_f = 2 * sa_p * sa_r / (sa_p + sa_r)

                aaa = 'Separate Tagging    P: {},  R:{},T F-measure: {}'.format(test_p, test_r, test_f)
                bbb = 'Separate Sentiment  P: {},  R:{},S F-measure: {}'.format(sa_p, sa_r, sa_f)
                ccc = 'Joint Tag&Sentiment P: {},  R:{},J F-measure: {}'.format(test_joint_senti_p, test_joint_senti_r,
                                                                                test_joint_senti_f)
                ddd = 'Joint Tag&Sentiment P: {},  R:{},C F-measure: {}'.format(test_joint_senti_p1,
                                                                                test_joint_senti_r1,
                                                                                test_joint_senti_f1)
                test_result.append(aaa)
                senti_test_result.append(bbb)
                print(aaa)
                print(bbb)
                print(ccc)
                print(ddd)

                if test_f > max_test_f:
                    max_test_f = test_f
                    fff = open(
                        './predict/' + data + str(test_file_id) + '_predict.' + model.model_name + 'use-reg-' + str(
                            args.use_reg), 'w')
                    for pred_, true_, text_ in zip(ori_y_pred, ori_y_true, ori_text):
                        fff.write('Text: ' + str(d.reverse(text_, d.text_dict_reverse)) + '\n')
                        fff.write('True: ' + str(true_) + '\n')
                        fff.write('Pred: ' + str(pred_) + '\n\n')
                    fff.close()
                if sa_f > max_senti_test_f:
                    max_senti_test_f = sa_f
                    fff = open('./predict/' + data + str(
                        test_file_id) + '_senti_predict' + model.model_name + 'use-reg-' + str(args.use_reg), 'w')
                    for pred_, true_, text_ in zip(ori_senti_y_pred, ori_senti_y_true, ori_text):
                        fff.write('Text: ' + str(d.reverse(text_, d.text_dict_reverse)) + '\n')
                        fff.write('True: ' + str(true_) + '\n')
                        fff.write('Pred: ' + str(pred_) + '\n\n')
                    fff.close()
            d.shuffle()
