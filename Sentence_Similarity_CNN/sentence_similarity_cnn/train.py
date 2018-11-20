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
