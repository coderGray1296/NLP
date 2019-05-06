import tensorflow as tf
t1 = [[[1, 2], [2, 3], [5, 6]], [[4, 4], [5, 3], [7, 8]]]
result = tf.concat(t1, axis=1)
print(result.shape)
