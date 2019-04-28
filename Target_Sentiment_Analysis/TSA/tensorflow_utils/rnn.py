# encoding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn


def RNN(x, seq_len, cell, hidden_size, layer_num=1, bidirectional=False):
    print('cell: {}'.format(cell))
    if cell == 'lstm':
        # fw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1., state_is_tuple=True)
        fw_cell = rnn.LSTMCell(hidden_size, use_peepholes=True)
        # if layer_num > 1:
        #    fw_cell = []
        #    for _ in range(layer_num):
        #        #fw_cell.append(rnn.BasicLSTMCell(hidden_size, forget_bias=1., state_is_tuple=True))
        #        fw_cell.append(rnn.LSTMCell(hidden_size, use_peepholes=True))
        if bidirectional:
            # bw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1., state_is_tuple=True)
            bw_cell = rnn.LSTMCell(hidden_size, use_peepholes=True)
            if layer_num > 1:
                bw_cell = []
                for _ in range(layer_num):
                    # bw_cell.append(rnn.BasicLSTMCell(hidden_size, forget_bias=1., state_is_tuple=True))
                    bw_cell.append(rnn.LSTMCell(hidden_size, use_peepholes=True))
                fw_cell = []
                for _ in range(layer_num):
                    # fw_cell.append(rnn.BasicLSTMCell(hidden_size, forget_bias=1., state_is_tuple=True))
                    fw_cell.append(rnn.LSTMCell(hidden_size, use_peepholes=True))
    elif cell == 'gru':
        fw_cell = rnn.GRUCell(hidden_size)
        # if layer_num > 1:
        #    fw_cell = []
        #    for _ in range(layer_num):
        #        fw_cell.append(rnn.GRUCell(hidden_size))
        if bidirectional:
            bw_cell = rnn.GRUCell(hidden_size)
            if layer_num > 1:
                bw_cell = []
                for _ in range(layer_num):
                    bw_cell.append(rnn.GRUCell(hidden_size))
                fw_cell = []
                for _ in range(layer_num):
                    fw_cell.append(rnn.GRUCell(hidden_size))

    else:
        raise ValueError('No match cell (lstm /gru)')

    if bidirectional:
        if layer_num > 1:
            rnn_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=x,
                                                                               sequence_length=seq_len,
                                                                               dtype=tf.float32)
        else:
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=x, sequence_length=seq_len,
                                                             dtype=tf.float32)
    else:
        rnn_outputs, _ = tf.nn.dynamic_rnn(fw_cell, inputs=x, sequence_length=seq_len, dtype=tf.float32)
    return rnn_outputs
