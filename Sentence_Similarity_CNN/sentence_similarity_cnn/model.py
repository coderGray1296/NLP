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
        