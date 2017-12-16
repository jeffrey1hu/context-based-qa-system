
import time
import logging
from os.path import join as pjoin

import numpy as np
import tensorflow as tf

from encoder import Encoder
from decoder import Decoder


class QASystem(object):
    def __init__(self, num_hidden, embed_size, embed_path, context_max_len, question_max_len, reg, keep_prob):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.num_hidden = num_hidden
        self.embed_size = embed_size
        self.embed_path = embed_path

        self.regularizer = tf.contrib.layers.l2_regularizer(reg)
        self.keep_prob = keep_prob


        # Place holder for context, question and answer
        self.context = tf.placeholder(tf.int32, shape=(None, context_max_len))
        self.context_m = tf.placeholder(tf.bool, shape=(None, context_max_len))

        self.question = tf.placeholder(tf.int32, shape=(None, question_max_len))
        self.question_m = tf.placeholder(tf.bool, shape=(None, question_max_len))

        self.answer_s = tf.placeholder(tf.int32, shape=(None,))
        self.answer_e = tf.placeholder(tf.int32, shape=(None,))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()



    def setup_system(self):
        H_r = Encoder(size=2 * self.num_hidden, keep_prob=self.keep_prob, num_hidden=self.num_hidden)\
            .encode(self.context, self.context_m, self.question, self.question_m, self.embedding)

        self.s_score, self.e_score = Decoder(2 * self.num_hidden, self.num_hidden, self.keep_prob, self.regularizer)\
            .decode(H_r, self.context_m)


    def setup_loss(self):
        with tf.variable_scope("loss"):
            loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_s, logits=self.s_score)
            loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_e, logits=self.e_score)

            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)

        self.final_loss = tf.reduce_mean(loss_e + loss_s) + reg_term
        tf.summary.scalar('final_loss', self.final_loss)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        logging.info('embed size: {} for path {}'.format(self.embed_size, self.embed_path))
        with tf.variable_scope("embeddings"):
            self.embedding = np.load(self.embed_path)['glove']
            self.embedding = tf.Variable(self.embedding, dtype=tf.float32, trainable=False)

