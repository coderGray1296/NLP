# encoding=utf-8
import tensorflow as tf

tf.set_random_seed(0)


def xavier(shape, n_in, n_out, name='W'):
    bound = tf.sqrt(6.0 / (n_in + n_out))
    return tf.random_uniform(shape, -bound, bound, name=name, seed=0)


def CNN2D(x, filter_shape, in_num=300, out_num=300, strides=[1, 1, 1, 1], padding='VALID', activation=tf.nn.relu):
    '''
    x: batch_size * in_height * in_width * channel
    fileters: filter_height * filter_width * in_channel * out_channel
    '''
    input_shape = x.shape
    W = tf.Variable(xavier(filter_shape, in_num, out_num), name='Conv2D_filter')
    b = tf.Variable(tf.zeros([filter_shape[-1]]), name='Conv2D_bias')

    outputs = tf.add(tf.nn.conv2d(x, W, strides, padding), b)
    if activation != None:
        outputs = activation(outputs)
    return outputs, (W, b)


def Pool2D(x, ksize, strides=[1, 1, 1, 1], padding='VALID', pool=tf.nn.max_pool):
    return pool(x, ksize, strides, padding)
