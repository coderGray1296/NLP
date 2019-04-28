# encoding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow_utils.rnn import RNN
from tensorflow_utils.utils import xavier, Linear
from tensorflow_utils.cnn import CNN2D, Pool2D
from tensorflow_utils.attention import attention
from regularization import calculate_reg


class Model(object):

    def __init__(self, args):

        self.model_name = 'pipeline'
        self.args = args

        self.text = tf.placeholder(tf.int32, [None, args.max_len], name='text')
        self.mask = tf.placeholder(tf.float32, [None, args.max_len], name='mask')
        self.pos = tf.placeholder(tf.int32, [None, args.max_len], name='pos')
        self.ner = tf.placeholder(tf.int32, [None, args.max_len], name='ner')
        self.label = tf.placeholder(tf.int32, [None, args.max_len], name='label')
        self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        self.char_len = tf.placeholder(tf.int32, [None, args.max_len], name='char_len')
        self.char = tf.placeholder(tf.int32, [None, args.max_len, args.max_character_len], name='char')
        self.context = tf.placeholder(tf.int32, [None, args.max_len, args.window_size], name='context')
        self.senti_label = tf.placeholder(tf.int32, [None, args.max_len], name='senti_label')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

        with tf.name_scope('Embedding'):
            if type(args.embedding) == type(None):
                word_E = tf.Variable(xavier([args.vocab_size, args.word_embedding_size], args.word_embedding_size,
                                            args.word_embedding_size), name='word_Embeddings')
            else:
                word_E = tf.Variable(args.embedding, name='word_Embeddings')

            if type(args.char_embedding) == type(None):
                char_E = tf.Variable(xavier([args.char_size, args.char_embedding_size], args.char_embedding_size,
                                            args.char_embedding_size), name='char_Embeddings')
            else:
                char_E = tf.Variable(args.char_embedding, name='char_embeddings')

            pos_E = tf.Variable(
                xavier([args.pos_size, args.pos_embedding_size], args.pos_embedding_size, args.pos_embedding_size),
                name='pos_Embeddings')
            ner_E = tf.Variable(
                xavier([args.ner_size, args.ner_embedding_size], args.ner_embedding_size, args.ner_embedding_size),
                name='ner_Embeddings')

            # word embeddings
            word_emb = tf.nn.embedding_lookup(word_E, self.text)
            # char embeddings
            char_emb = tf.nn.embedding_lookup(char_E, self.char)
            char_emb = tf.reshape(char_emb, [-1, args.max_character_len, args.char_embedding_size])
            char_len = tf.reshape(self.char_len, [-1])
            # pos embeddings
            pos_emb = tf.nn.embedding_lookup(pos_E, self.pos)
            # ner embeddings
            ner_emb = tf.nn.embedding_lookup(ner_E, self.ner)
            # context embeddings
            context_emb = tf.nn.embedding_lookup(word_E, self.context)
            # context_emb = tf.transpose(context_emb, perm=[0,1,3,2])

            if args.dropout:
                pos_emb = tf.nn.dropout(pos_emb, self.dropout_rate, seed=0)
                ner_emb = tf.nn.dropout(ner_emb, self.dropout_rate, seed=0)
                word_emb = tf.nn.dropout(word_emb, self.dropout_rate, seed=0)
                char_emb = tf.nn.dropout(char_emb, self.dropout_rate, seed=0)
                context_emb = tf.nn.dropout(context_emb, self.dropout_rate, seed=0)

        # cnn begin
        cnn_input = context_emb
        # filter_shape = [1, args.word_embedding_size, args.window_size, args.cnn_kernels]
        filter_shape = [1, args.window_size, args.word_embedding_size, args.cnn_kernels]
        cnn_outputs, _ = CNN2D(cnn_input, filter_shape)
        cnn_outputs = tf.reshape(cnn_outputs, [-1, args.cnn_kernels], name='cnn_outputs')
        # cnn end

        # char rnn begin
        with tf.variable_scope('char_level_lstm'):
            char_rnn_outputs = RNN(char_emb, char_len, args.cell, args.char_hidden_size, args.rnn_layer_num,
                                   args.bi_rnn)
        rnn_dim = args.char_hidden_size
        if args.bi_rnn:
            char_rnn_outputs = tf.concat(char_rnn_outputs, axis=2)
            rnn_dim *= 2
        char_rnn_outputs = tf.reshape(char_rnn_outputs, [-1, args.max_len, args.max_character_len, rnn_dim])
        char_rnn_outputs = tf.reduce_max(char_rnn_outputs, axis=2)
        # char rnn end

        # rnn begin
        feature_num = args.hidden_size + rnn_dim + args.pos_embedding_size + args.ner_embedding_size + 100
        with tf.variable_scope('word_level_lstm'):
            # rnn_inputs  = tf.concat([word_emb, char_rnn_outputs], axis=2)
            rnn_inputs = word_emb  # tf.concat([word_emb, char_rnn_outputs], axis=2)
            rnn_outputs = RNN(rnn_inputs, self.seq_len, args.cell, feature_num, args.rnn_layer_num, args.bi_rnn)
        if args.bi_rnn:
            rnn_outputs = tf.concat(rnn_outputs, axis=2)
            feature_num *= 2
            senti_feature_num = feature_num
        # rnn end

        # attention begin
        attention_inputs = rnn_outputs * tf.expand_dims(self.mask, -1)  # masked

        attention_vector = tf.reshape(attention_inputs, [-1, feature_num])
        attention_vector = tf.stack([attention_vector] * args.max_len)
        attention_vector = tf.transpose(attention_vector, perm=[1, 0, 2])

        attention_inputs = tf.stack([attention_inputs] * args.max_len)  # for each word
        attention_inputs = tf.transpose(attention_inputs, perm=[1, 0, 2, 3])
        attention_inputs = tf.reshape(attention_inputs, [-1, args.max_len, feature_num])

        attention_vector = tf.concat([attention_inputs, attention_vector], axis=2)

        attention_outputs = attention(attention_inputs, attention_vector, feature_num)
        rnn_outputs = tf.reshape(rnn_outputs, [-1, feature_num], name='rnn_outputs')
        # attention end

        if args.use_cnn:
            self.outputs = tf.concat([rnn_outputs, cnn_outputs], axis=1)
            self.senti_outputs = tf.concat([rnn_outputs, attention_outputs], axis=1)
            feature_num += args.cnn_kernels
            senti_feature_num *= 2
        else:
            self.outputs = rnn_outputs
            self.senti_outputs = rnn_outputs
        if args.dropout:
            self.outputs = tf.nn.dropout(self.outputs, self.dropout_rate, seed=0)
            self.senti_outputs = tf.nn.dropout(self.senti_outputs, self.dropout_rate, seed=0)

        mlp_outputs, _ = Linear(self.outputs, feature_num, feature_num, activation=tf.nn.relu)
        senti_mlp_outputs, _ = Linear(self.senti_outputs, senti_feature_num, senti_feature_num, activation=tf.nn.relu)
        activation = tf.nn.softmax
        if args.use_crf:
            activation = None

        # pipeline
        self.logits1, _ = Linear(mlp_outputs, feature_num, args.label_size, activation=activation)
        concatenation = tf.concat([self.logits1, senti_mlp_outputs], axis=1)
        self.logits2, _ = Linear(concatenation, senti_feature_num + args.label_size, args.senti_label_size,
                                 activation=activation)

        y_pred = tf.argmax(self.logits1, axis=1)
        self.y_pred = tf.reshape(y_pred, [-1, args.max_len], name='y_pred')

        senti_y_pred = tf.argmax(self.logits2, axis=1)
        self.senti_y_pred = tf.reshape(senti_y_pred, [-1, args.max_len], name='senti_y_pred')

        self.logits = tf.reshape(self.logits1, [-1, args.max_len, args.label_size], name='logits')
        self.senti_logits = tf.reshape(self.logits2, [-1, args.max_len, args.senti_label_size], name='senti_logits')

        self.loss = self.compute_loss()

        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, use_locking=True)
        grads_and_vars = opt.compute_gradients(self.loss)
        grads_and_vars = [(g, v) if g is None else (tf.clip_by_norm(g, args.max_grad_norm), v) for g, v in
                          grads_and_vars]
        self.train_op = opt.apply_gradients(grads_and_vars)
        # self.train_op = opt.minimize(self.loss)

    def predict_batch(self, sess, fd):

        if self.args.use_crf:
            logits, seq_len, transition_params = sess.run([self.logits, self.seq_len, self.transition_params],
                                                          feed_dict=fd)
        else:
            logits, seq_len = sess.run([self.logits, self.seq_len], feed_dict=fd)
            transition_params = np.asarray([[1.0, 1.0, -100], [1.0, 1.0, 1.0], [1.0, 1.0, -100.0]])
        result = []
        for l, s in zip(logits, seq_len):
            y_pred, _ = tf.contrib.crf.viterbi_decode(l[:s], transition_params)
            y_pred = list(y_pred) + [0] * (self.args.max_len - s)
            result.append(y_pred)
        return np.asarray(result)

    def predict_batch_senti(self, sess, fd):

        if self.args.use_crf:
            logits, seq_len, transition_params = sess.run(
                [self.senti_logits, self.seq_len, self.senti_transition_params], feed_dict=fd)
        else:
            logits, seq_len = sess.run([self.logits, self.seq_len], feed_dict=fd)
            transition_params = np.asarray([[1.0, 1.0, -100], [1.0, 1.0, 1.0], [1.0, 1.0, -100.0]])
        result = []
        for l, s in zip(logits, seq_len):
            y_pred, _ = tf.contrib.crf.viterbi_decode(l[:s], transition_params)
            y_pred = list(y_pred) + [0] * (self.args.max_len - s)
            result.append(y_pred)
        return np.asarray(result)

    def compute_loss(self):

        if self.args.use_crf:
            with tf.variable_scope('target_detect_loss'):
                log_likehood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.label,
                                                                                         self.seq_len)
                target_loss = tf.reduce_mean(-log_likehood)

            with tf.variable_scope('sentiment_loss'):
                senti_log_likehood, self.senti_transition_params = tf.contrib.crf.crf_log_likelihood(self.senti_logits,
                                                                                                     self.senti_label,
                                                                                                     self.seq_len)
                senti_loss = tf.reduce_mean(-senti_log_likehood)

            loss = target_loss + senti_loss
        else:
            labels = tf.reshape(tf.contrib.layers.one_hot_encoding(tf.reshape(self.label, [-1]),
                                                                   num_classes=self.args.label_size),
                                [-1, self.args.max_len, self.args.label_size])
            cross_entropy = -tf.reduce_sum(labels * tf.log(self.logits1), axis=2) * self.mask
            cross_entropy = tf.reduce_sum(cross_entropy, axis=1) / tf.cast(self.seq_len, tf.float32)
            target_loss = tf.reduce_mean(cross_entropy)

            senti_labels = tf.reshape(tf.contrib.layers.one_hot_encoding(tf.reshape(self.senti_label, [-1]),
                                                                         num_classes=self.args.senti_label_size),
                                      [-1, self.args.max_len, self.args.senti_label_size])
            senti_cross_entropy = -tf.reduce_sum(senti_labels * tf.log(self.logits2), axis=2) * self.mask
            senti_cross_entropy = tf.reduce_sum(senti_cross_entropy, axis=1) / tf.cast(self.seq_len, tf.float32)
            senti_loss = tf.reduce_mean(senti_cross_entropy)
            loss = target_loss + senti_loss
        '''
        if self.args.l2_coefficient != 0.0:
            for v in tf.trainable_variables():
                loss += self.args.l2_coefficient * tf.reduce_sum(tf.nn.l2_loss(v))
        '''
        if self.args.use_reg:
            reg = calculate_reg(tf.nn.softmax(self.logits), tf.nn.softmax(self.senti_logits), self.mask, self.seq_len)
            loss += reg
        return loss
