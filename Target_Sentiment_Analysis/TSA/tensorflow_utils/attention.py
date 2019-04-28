import tensorflow as tf


def attention(lstm_inputs, attention_inputs, attention_size):
    inputs_shape = attention_inputs.shape
    sequence_length = inputs_shape[1].value
    hidden_size = inputs_shape[2].value

    W = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(attention_inputs, [-1, hidden_size]), W) + tf.reshape(b, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u, [-1, 1]))
    exp = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alpha = exp / tf.reshape(tf.reduce_sum(exp, 1), [-1, 1])

    output = tf.reduce_sum(lstm_inputs * tf.reshape(alpha, [-1, sequence_length, 1]), 1)
    return output
