# encoding=utf-8
import tensorflow as tf


def calculate_reg(sentiment_logits, entity_logits, mask, sent_len):
    sent_O = sentiment_logits[:, :, 0]
    entity_O = entity_logits[:, :, 0]

    sent_B = (sentiment_logits[:, :, 1] + sentiment_logits[:, :, 3]) / 2
    entity_B = (entity_logits[:, :, 1] + entity_logits[:, :, 3] + entity_logits[:, :, 5]) / 3

    sent_I = (sentiment_logits[:, :, 2] + sentiment_logits[:, :, 4]) / 2
    entity_I = (entity_logits[:, :, 2] + entity_logits[:, :, 4] + entity_logits[:, :, 6]) / 3

    O_l2 = tf.reduce_sum(tf.nn.l2_loss(sent_O - entity_O) * mask, axis=1) / tf.cast(sent_len, tf.float32)
    B_l2 = tf.reduce_sum(tf.nn.l2_loss(sent_B - entity_B) * mask, axis=1) / tf.cast(sent_len, tf.float32)
    I_l2 = tf.reduce_sum(tf.nn.l2_loss(sent_I - entity_I) * mask, axis=1) / tf.cast(sent_len, tf.float32)

    l2_reg = tf.reduce_mean(O_l2)  # + tf.reduce_mean(B_l2) + tf.reduce_mean(I_l2)

    return l2_reg
