
import tensorflow as tf
from utils import variable_summaries

def softmax_mask_prepro(tensor, mask):
    # set huge neg number(-1e10) in padding area
    assert tensor.get_shape().ndims == mask.get_shape().ndims
    m0 = tf.subtract(tf.constant(1.0), tf.cast(mask, 'float32'))
    paddings = tf.multiply(m0, tf.constant(-1e10))
    tensor = tf.where(mask, tensor, paddings)
    return tensor


class Decoder(object):
    def __init__(self, output_size, num_hidden, keep_prob, regularizer):
        self.output_size = output_size
        self.num_hidden = num_hidden
        self.regularizer = regularizer
        self.keep_prob = keep_prob

    def decode(self, H_r, context_m):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        context_m_float = tf.cast(context_m, tf.float32)
        # shape -> (b, q, 4n)
        H_r_shape = tf.shape(H_r)
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("decoder"):
            W_r = tf.get_variable("V_r", shape=[self.num_hidden * 4, self.num_hidden * 2],
                                  dtype=tf.float32, initializer=initializer, regularizer=self.regularizer)

            W_f = tf.get_variable("W_f", shape=[self.num_hidden * 2, 1],
                                  dtype=tf.float32, initializer=initializer, regularizer=self.regularizer)

            W_h = tf.get_variable("W_h", shape=[self.num_hidden * 4, self.num_hidden * 2],
                                  dtype=tf.float32, initializer=initializer, regularizer=self.regularizer)

            B_r = tf.get_variable("B_r", shape=[self.num_hidden * 2], dtype=tf.float32, initializer=tf.zeros_initializer())
            B_f = tf.get_variable("B_f", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer())

            W_r_e = tf.tile(tf.expand_dims(W_r, axis=0), multiples=[H_r_shape[0], 1, 1])
            W_f_e = tf.tile(tf.expand_dims(W_f, axis=0), multiples=[H_r_shape[0], 1, 1])
            # W_h_e = tf.tile(tf.expand_dims(W_h, axis=0), multiples=[H_r_shape[0], 1, 1])

            # f1 -> (b, q, 2n)
            f1 = tf.nn.tanh(tf.matmul(H_r, W_r_e) + B_r)
            f1 = tf.nn.dropout(f1, keep_prob=self.keep_prob)
            with tf.name_scope('starter_score'):
                # s_score -> (b, q, 1)
                s_score = tf.matmul(f1, W_f_e) + B_f
                # s_score -> (b, q)
                s_score = tf.squeeze(s_score, axis=2)
                s_score = softmax_mask_prepro(s_score, context_m)
                variable_summaries(s_score)

            with tf.name_scope('starter_prob'):
                # the prob distribution of start index
                s_prob = tf.nn.softmax(s_score)
                s_prob = s_prob * context_m_float
                variable_summaries(s_prob)
            # Hr_attend -> (batch_size, 4n)
            Hr_attend = tf.reduce_sum(H_r * tf.expand_dims(s_prob, axis=2), axis=1)

            # f2 = tf.nn.tanh(tf.matmul(H_r, W_r_e)
            #                 + tf.matmul(tf.tile(tf.expand_dims(Hr_attend, axis=1), multiples=[1, H_r_shape[1], 1]), W_h_e)
            #                 + B_r)
            f2 = tf.nn.tanh(tf.matmul(H_r, W_r_e)
                            + tf.expand_dims(tf.matmul(Hr_attend, W_h), axis=1)
                            + B_r)
            with tf.name_scope('end_score'):
                e_score = tf.matmul(f2, W_f_e) + B_f
                e_score = tf.squeeze(e_score, axis=2)
                e_score = softmax_mask_prepro(e_score, context_m)
                variable_summaries(e_score)

            with tf.name_scope('end_prob'):
                e_prob = tf.nn.softmax(e_score)
                e_prob = tf.multiply(e_prob, context_m_float)
                variable_summaries(e_prob)

        return s_score, e_score
