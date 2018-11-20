import datetime
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import preprosessing
from model import CNN

train_sample_percentage = 0.9
data_file = 'example.txt'
filter_sizes = [3, 4, 5]
num_filters = 128
#sequence_length = 36
seq_length = 36
num_classes = 1
dropout_keep_prob = 0.5
l2_reg_lambda = 1

batch_size = 64
num_epochs = 200
evaluate_every = 100
checkpoints_every = 100
num_checkpoints = 5
allow_soft_placement = True
log_device_placement = False

print('loading data...')
s1, s2, score = preprosessing.read_data_sets(data_file)
#change type of score into np.array
score = np.asarray([s] for s in score)
sample_num = len(score)
train_end = int(sample_num * train_sample_percentage)

#split train and test dataset
s1_train, s1_test = s1[:train_end], s1[train_end:]
s2_train, s2_test = s2[:train_end], s2[train_end:]
score_train, score_test = score[:train_end], score[train_end:]

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = allow_soft_placement,
        log_device_placement = log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN(
            squence_length = seq_length,
            num_classes = num_classes,
            filter_sizes = filter_sizes,
            num_filters = num_filters,
            l2_reg_lambda = l2_reg_lambda
        )

    #define training procedure
    globals_step = tf.Variable(0, name='globals_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)

    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=globals_step)

    #output dictories for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, 'run', timestamp))
    print('Writting to {}\n'.format(out_dir))

    #Summaries for loss and pearson
    loss_summary = tf.summary.scalar('loss', cnn.loss)
    accuracy_summary = tf.summary.scalar('pearson', cnn.pearson)

    #Training Summaries
    train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
    train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    #Testing Summaries
    test_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
    test_summary_dir = os.path.join(out_dir, 'summaries', 'test')
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

    #checkpoints saver
    checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, 'checkpoints'))
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
    if not os._exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

    #Initial all variables
    sess.run(tf.global_variables_initializer)
    sess.run(tf.local_variables_initializer)


    def train_step(s1, s2, score):

