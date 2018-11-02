import tensorflow as tf
import numpy as np
import time
import datetime
import os
from text_cnn import TextCNN
import preprocessing
from tensorflow.contrib import learn

#Percentage of the training data to use for validation
dev_sample_percentage = .1
positive_data_file = './data/positive.txt'
negative_data_file = './data/negative.txt'

#Dimensionality of character embedding (default: 128)
embedding_dim = 128
#Comma-separated filter sizes (default: '3,4,5')
filter_sizes = [3,4,5]
#Number of filters per filter size (default: 128)
num_filters = 128
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0

batch_size = 32
num_epochs = 200
#Evaluate model on dev set after this many steps (default: 100)
evaluate_every = 100
#Save model after this many steps (default: 100)
checkpoint_every = 100
#Number of checkpoints to store (default: 5)
num_checkpoints = 5

allow_soft_placement = True
log_device_placement = False

#Data Preparation
print('Loading data...')
x_text, y = preprocessing.load_data(positive_data_file, negative_data_file)
# Build vocabulary
#计算每行数据词数的最大长度
max_document_length = max([len(x.split(" ")) for x in x_text])
#print(max_document_length)
#进行词表填充，同时将词转化为索引序号
vocab = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab.fit_transform(x_text)))

#Random shuffle data
np.random.seed(10)
# np.arange生成随机序列
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffle = x[shuffle_indices]
y_shuffle = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
#print(dev_sample_index)
x_train, x_test = x_shuffle[:dev_sample_index], x_shuffle[dev_sample_index:]
y_train, y_test = y_shuffle[:dev_sample_index], y_shuffle[dev_sample_index:]
#print("Vocabulary Size: {:d}".format(len(vocab.vocabulary_)))
#print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))

#Training
#===============================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = allow_soft_placement,
        log_device_placement = log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            
        )






