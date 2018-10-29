import tensorflow as tf
import numpy as np

class TextCNN(object):
    def __init__(self, input_size, num_classes, vocab_size, embedding_size, filter_sizes):

        #placeholders for input,output,dropout
        self.input_x = tf.placeholder(tf.int32, [None, input_size], name="input")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="output")
        self.dropout_keep_prob = tf.placeholder

        #keep track of l2 regularzation
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        #creating convolution + maxpool for ever layer
