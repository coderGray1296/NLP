# encoding=utf-8
import tensorflow as tf

tf.set_random_seed(0)


def xavier(shape, n_in, n_out, name='W'):
    bound = tf.sqrt(6.0 / (n_in + n_out))
    return tf.random_uniform(shape, -bound, bound, name=name, seed=0)


def Linear(x, input_size, output_size, name='Linear', activation=tf.tanh, initializer=xavier):
    if initializer == xavier:
        W = tf.Variable(initializer([input_size, output_size], input_size, output_size), name=name + '_W')
        b = tf.Variable(initializer([output_size], input_size, output_size), name=name + '_b')
    else:
        W = tf.Variable(initializer([input_size, output_size]), name=name + '_W')
        b = tf.Variable(initializer([output_size]), name=name + '_b')

    outputs = tf.add(tf.matmul(x, W), b)
    if activation != None:
        outputs = activation(outputs)
    return outputs, (W, b)
