
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from utils import *


def sequence_length(sequence_mask):
    """
    Args:
        sequence_mask: Bool tensor with shape -> [batch_size, q]

    Returns:
        tf.int32, [batch_size]

    """
    return tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)


class Encoder(object):
    def __init__(self, size, keep_prob, num_hidden):
        self.size = size
        self.num_hidden = num_hidden
        self.keep_prob = keep_prob

    def encode(self, context, context_m, question, question_m, embedding):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        # context shape -> (None, P)
        # context embed -> (None, P, n)
        context_embed = tf.nn.embedding_lookup(embedding, context)
        context_embed = tf.nn.dropout(context_embed, keep_prob=self.keep_prob)
        question_embed = tf.nn.embedding_lookup(embedding, question)
        question_embed = tf.nn.dropout(question_embed, keep_prob=self.keep_prob)

        with tf.variable_scope('context_lstm'):
            con_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
            con_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
            # shape of outputs -> [output_fw, output_bw] -> output_fw -> [batch_size, P, n]
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(con_lstm_fw_cell,
                                                         con_lstm_bw_cell,
                                                         context_embed,
                                                         sequence_length=sequence_length(context_m),
                                                         dtype=tf.float32)

        # need H_context as dim (Batch_size, hidden_size, P)
        # dimension of outputs
        with tf.variable_scope('H_context'):
            # H_context -> (batch_size, P, 2n)
            H_context = tf.concat(outputs, axis=2)
            H_context = tf.nn.dropout(H_context, keep_prob=self.keep_prob)
            variable_summaries(H_context)

        with tf.variable_scope('question_lstm'):
            question_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
            question_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
            # shape of outputs -> [output_fw, output_bw] -> output_fw -> [batch_size, P, n]
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(question_lstm_fw_cell,
                                                         question_lstm_bw_cell,
                                                         question_embed,
                                                         sequence_length=sequence_length(question_m),
                                                         dtype=tf.float32)

        with tf.variable_scope('H_question'):
            # H_question -> (batch_size, Q, 2n)
            H_question = tf.concat(outputs, axis=2)
            H_question = tf.nn.dropout(H_question, keep_prob=self.keep_prob)
            variable_summaries(H_question)


        with tf.variable_scope('H_match_lstm'):
            match_lstm_fw_cell = matchLSTMcell(2 * self.num_hidden, self.size, H_question, question_m)
            match_lstm_bw_cell = matchLSTMcell(2 * self.num_hidden, self.size, H_question, question_m)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(match_lstm_fw_cell,
                                                         match_lstm_bw_cell,
                                                         H_context,
                                                         sequence_length=sequence_length(context_m),
                                                         dtype=tf.float32)

        # H_match -> (batch_size, Q, 2n)
        with tf.variable_scope('H_match'):
            H_r = tf.concat(outputs, axis=2)
            H_r = tf.nn.dropout(H_r, keep_prob=self.keep_prob)
            variable_summaries(H_r)

        return H_r


class matchLSTMcell(RNNCell):

    def __init__(self, input_size, state_size, h_question, h_question_m):
        self.input_size = input_size
        self._state_size = state_size
        self.h_question = h_question
        self.h_question_m = tf.cast(h_question_m, tf.float32)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):

        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            # the batch size
            example_num = tf.shape(inputs)[0]

            # TODO: figure out the right way to initialize rnn weights.
            initializer = tf.contrib.layers.xavier_initializer()
            # initializer = tf.uniform_unit_scaling_initializer(1.0)
            # tf.name_scope()
            w_q = tf.get_variable("W_q", shape=[self.input_size, self.input_size], dtype=tf.float32, initializer=initializer)
            w_p = tf.get_variable("W_p", shape=[self.input_size, self.input_size], dtype=tf.float32, initializer=initializer)
            w_r = tf.get_variable("W_r", shape=[self.state_size, self.input_size], dtype=tf.float32, initializer=initializer)
            w_a = tf.get_variable("W_a", shape=[self.input_size, 1], dtype=tf.float32, initializer=initializer)

            b_p = tf.get_variable("b_p", shape=[self.input_size], initializer=tf.zeros_initializer(), dtype=tf.float32)
            b_a = tf.get_variable("b_a", shape=[1], initializer=tf.zeros_initializer(), dtype=tf.float32)

            # w_q_e -> b * 2n * 2n
            w_q_e = tf.tile(tf.expand_dims(w_q, axis=0), [example_num, 1, 1])

            # g -> b * q * 2n
            g = tf.nn.tanh(tf.matmul(self.h_question, w_q_e)  # shape b * q * 2n
                             + tf.expand_dims(tf.matmul(inputs, w_p) + tf.matmul(state, w_r) + b_p, axis=1)
                             )

            w_a_e = tf.tile(tf.expand_dims(w_a, axis=0), [example_num, 1, 1])

            # alpha -> b * q
            alpha = tf.nn.softmax(tf.squeeze(tf.matmul(g, w_a_e) # shape b * q * 1
                                  + b_a, axis=[2]))
            # mask out the attention over the padding.
            alpha = alpha * self.h_question_m

            # question_attend -> b * 2n
            question_attend = tf.reduce_sum((tf.expand_dims(alpha, [2]) * self.h_question), axis=1)
            # z -> b * 4n
            z = tf.concat([inputs, question_attend], axis=1)

            # with GRU instead
            W_r_gru = tf.get_variable("W_r_gru",
                                      shape=[self._state_size, self._state_size],
                                      dtype=tf.float32,
                                      initializer=initializer)
            U_r_gru = tf.get_variable("U_r_gru",
                                      shape=[self.input_size * 2, self._state_size],
                                      dtype=tf.float32,
                                      initializer=initializer)

            # initialize b_r as 1.0 for default "remember"
            b_r_gru = tf.get_variable("b_r_gru",
                                      shape=[self._state_size],
                                      dtype=tf.float32,
                                      initializer=tf.constant_initializer(1.0))

            W_z_gru = tf.get_variable("W_z_gru",
                                      shape=[self._state_size, self._state_size],
                                      dtype=tf.float32,
                                      initializer=initializer)

            U_z_gru = tf.get_variable("U_z_gru",
                                      shape=[self.input_size * 2, self._state_size],
                                      dtype=tf.float32,
                                      initializer=initializer)

            # initialize b_z as 1.0 for default "remember"
            b_z_gru = tf.get_variable("b_z_gru",
                                  shape=[self._state_size],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(1.0))

            W_o_gru = tf.get_variable("W_o_gru",
                                      shape=[self._state_size, self._state_size],
                                      dtype=tf.float32,
                                      initializer=initializer)

            U_o_gru = tf.get_variable("U_o_gru",
                                      shape=[self.input_size * 2, self._state_size],
                                      dtype=tf.float32,
                                      initializer=initializer)

            b_o_gru = tf.get_variable("b_o_gru", shape=[self._state_size], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            z_t = tf.nn.sigmoid(tf.matmul(z, U_z_gru) + tf.matmul(state, W_z_gru) + b_z_gru)
            r_t = tf.nn.sigmoid(tf.matmul(z, U_r_gru) + tf.matmul(state, W_r_gru) + b_r_gru)

            h_hat = tf.nn.tanh(tf.matmul(z, U_o_gru) + r_t * tf.matmul(state, W_o_gru) + b_o_gru)
            output = z_t * state + (1 - z_t) * h_hat

            new_state = output
        return output, new_state
