import tensorflow as tf
import numpy as np
from preprocessing import build_glove_dic

class CNN(object):
    def __init__(self, sequence_length, num_classes, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.input_s1 = tf.placeholder(tf.int32, [None, sequence_length], name='input_s1')
        self.input_s2 = tf.placeholder(tf.int32, [None, sequence_length], name='input_s2')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = tf.constant(0.0)

        self.init_weight()
        self.inference()
        self.dropout()
        self.add_output()
        self.add_acc_loss()

    def init_weight(self):
        #Embedding layer
        _, self.word_embedding = build_glove_dic()
        self.embedding_size = self.word_embedding.shape[1]
        self.W = tf.get_variable(name="word_embedding", shape=self.embedding_size.shape, dtype=tf.float32, initializer=tf.constant_initializer(self.word_embedding), trainable=True)
        self.s1 = tf.nn.embedding_lookup(self.W, self.input_s1)
        self.s2 = tf.nn.embedding_lookup(self.W, self.input_s2)
        #concat s1 and s2 in a total vector in the second dim [None, sequence_length * 2, embedding_size, input_channel]
        self.x = tf.concat([self.s1, self.s2], axis=1)
        #expand dims in the last dimision for input channel
        self.x = tf.expand_dims(self.x, -1)

    def inference(self):
        #Create convolution layer and maxpool layer for each filter size
        pooled_output = []
        for i, filter_size in self.filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                #concolution layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.x,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv'
                )
                #apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #maxpool layer
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length * 2 - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool'
                )
                pooled_output.append(pooled)
        #combine all the pooled features
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_output, 3)
        self.h_pooled = tf.reshape(self.h_pool, [-1, self.num_filters_total])

    def dropout(self):
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pooled, self.dropout_keep_prob)

    def add_output(self):
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')

    def add_acc_loss(self):
        #Loss
        with tf.name_scope("loss"):
            losses = tf.square(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
        #Accuracy
        with tf.name_scope("pearson"):
            mid1 = tf.reduce_mean(self.scores * self.input_y) - \
                   tf.reduce_mean(self.scores) * tf.reduce_mean(self.input_y)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(self.input_y)) - tf.square(tf.reduce_mean(self.input_y)))
            self.pearson = mid1 / mid2

