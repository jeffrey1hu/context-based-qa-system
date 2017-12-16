
import sys
import time
import numpy as np
import logging
import tensorflow as tf
from os.path import join as pjoin
from tqdm import tqdm

from evaluate import f1_score, exact_match_score


class QaSystemSolver(object):
    def __init__(self, model, dataset, answers, raw_answers, rev_vocab, **kwargs):

        self.model = model
        self.dataset = dataset
        self.answers = answers
        self.raw_answers = raw_answers
        self.rev_vocab = rev_vocab


        self.batch_size = kwargs.pop('batch_size', 64)
        self.start_lr = kwargs.pop('start_lr', 0.0001)
        self.epochs = kwargs.pop('epochs', 5)

        self.start_steps = kwargs.pop('start_steps', 0)
        self.print_every = kwargs.pop('print_every', 1000)
        self.save_every = kwargs.pop('save_every', 1000)
        self.save_every_epoch = kwargs.pop('save_every_epoch', True)

        self.sample = kwargs.pop('sample', 0)
        self.debug_num = kwargs.pop('debug_num', 0)
        self.max_grad_norm = kwargs.pop('max_grad_norm', 5.0)

        self.opt = kwargs.pop('opt', 'adam')
        self.train_dir = kwargs.pop('train_dir', 'output/train')
        self.summary_dir = kwargs.pop('summary_dir', 'output/tensorboard/summary')

        if self.opt == "adam":
            self.optimizer = tf.train.AdamOptimizer

        self.global_step = tf.Variable(self.start_steps, trainable=False)
        self.starter_learning_rate = tf.placeholder(tf.float32, name='start_lr')

    def train(self):
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        # train_context -> (num, 2, max_length)
        train_context = np.array(self.dataset['train_context'])
        train_question = np.array(self.dataset['train_question'])
        # train_answer -> (num, 2)
        train_answer = np.array(self.answers['train_answer'])

        if self.debug_num:
            assert isinstance(self.debug_num, int), 'the debug number should be a integer'
            assert self.debug_num < len(train_answer), 'check debug number!'
            train_answer = train_answer[0:self.debug_num]
            train_context = train_context[0:self.debug_num]
            train_question = train_question[0:self.debug_num]
            print_every = 5

        num_example = len(train_answer)
        logging.info('num example is {}'.format(num_example))
        shuffle_list = np.arange(num_example)

        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.model.global_step,
                                                   1000, 0.9, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate)
            grad_var = optimizer.compute_gradients(self.model.final_loss)
            grad = [i[0] for i in grad_var]
            var = [i[1] for i in grad_var]
            grad_norm = tf.global_norm(grad)
            tf.summary.scalar('grad_norm', grad_norm)
            grad, use_norm = tf.clip_by_global_norm(grad, self.max_grad_norm)
            train_op = optimizer.apply_gradients(zip(grad, var), global_step=self.model.global_step)

        saver = tf.train.Saver()
        merged = tf.summary.merge_all()

        losses = []
        norms = []
        train_evals = []
        val_evals = []
        iters = self.start_steps
        save_path = pjoin(self.train_dir, 'weights')
        batch_size = self.batch_size
        batch_num = int(num_example * 1.0 / batch_size)
        total_iterations = self.epochs * batch_num + self.start_steps

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            train_writer = tf.summary.FileWriter(self.summary_dir + str(self.start_lr),
                                                              sess.graph)
            tic = time.time()

            for ep in range(self.epochs):
                np.random.shuffle(shuffle_list)
                train_context = train_context[shuffle_list]
                train_question = train_question[shuffle_list]
                train_answer = train_answer[shuffle_list]

                logging.info('training epoch ---- {}/{} -----'.format(ep + 1, self.epochs))
                ep_loss = 0.

                for it in xrange(batch_num):
                    sys.stdout.write('> %d / %d \r' % (iters % print_every, print_every))
                    sys.stdout.flush()

                    context = train_context[it * batch_size: (it + 1) * batch_size]
                    question = train_question[it * batch_size: (it + 1) * batch_size]
                    answer = train_answer[it * batch_size: (it + 1) * batch_size]

                    input_feed = self.load_input_feed(context, question, answer)
                    batch_final_loss = self.model.final_loss
                    summary, _,  loss, grad_norm= sess.run([merged, train_op, batch_final_loss, grad_norm], input_feed)

                    train_writer.add_summary(summary, iters)
                    ep_loss += loss
                    losses.append(loss)
                    norms.append(grad_norm)
                    iters += 1

                    if iters % print_every == 0:
                        toc = time.time()
                        logging.info('iters: {}/{} loss: {} norm: {}. time: {} secs'.format(
                            iters, total_iterations, loss, grad_norm, toc - tic))
                        tf1, tem, f1, em = self.evaluate_answer(sess, self.dataset, self.raw_answers, self.rev_vocab,
                                                                training=True, log=True, sample=self.sample)
                        train_evals.append((tf1, tem))
                        val_evals.append((f1, em))
                        tic = time.time()

                    if iters % self.save_every == 0:
                        saver.save(sess, save_path, global_step=iters)
                        self.evaluate_answer(sess, self.dataset, self.raw_answers, self.rev_vocab,
                                             training=True, log=True, sample=self.sample)

                if self.save_every_epoch:
                    saver.save(sess, save_path, global_step=iters)
                    self.evaluate_answer(sess, self.dataset, self.raw_answers, self.rev_vocab,
                                         training=True, log=True, sample=4000)
                logging.info('average loss of epoch {}/{} is {}'.format(ep + 1, self.epochs, ep_loss / batch_num))

                data_dict = {'losses': losses, 'norms': norms,
                             'train_eval': train_evals, 'val_eval': val_evals}
                c_time = time.strftime('%Y%m%d_%H%M', time.localtime())
                data_save_path = pjoin('cache', str(iters) + 'iters' + c_time + '.npz')
                np.savez(data_save_path, data_dict)

    def evaluate_answer(self, sess, dataset, raw_answers, rev_vocab,
                        sample=(100, 100), log=False, training=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        if not isinstance(rev_vocab, np.ndarray):
            rev_vocab = np.array(rev_vocab)

        if not isinstance(sample, tuple):
            sample = (sample, sample)

        tf1 = 0.
        tem = 0.

        input_batch_size = 100

        if training:
            train_len = sample[0]

            train_context = dataset['train_context'][:train_len]
            train_question = dataset['train_question'][:train_len]
            train_answer = raw_answers['raw_train_answer'][:train_len]


            train_a_s = np.array([], dtype=np.int32)
            train_a_e = np.array([], dtype=np.int32)

            for i in tqdm(range(train_len // input_batch_size), desc='trianing set'):
                train_as, train_ae = self.answer(self.model, sess,
                                                 np.array(train_context[i * input_batch_size:(i + 1) * input_batch_size]),
                                                 np.array(train_question[i * input_batch_size:(i + 1) * input_batch_size]))

                train_a_s = np.concatenate((train_a_s, train_as), axis=0)
                train_a_e = np.concatenate((train_a_e, train_ae), axis=0)

            # a_s and a_e -> (sample_num)
            for i in range(train_len):
                prediction_ids = train_context[i][0][train_a_s[i]:train_a_e[i]+1]
                prediction_answer = ' '.join(rev_vocab[prediction_ids])
                raw_answer = train_answer[i]
                tf1 += f1_score(prediction_answer, raw_answer)
                tem += exact_match_score(prediction_answer, raw_answer)
                # if i < 10:
                #     print("predict_answer: ", prediction_answer)
                #     print("ground truth: ", raw_answer)
                #     print ("f1: ", f1_score(prediction_answer, raw_answer))

            if log:
                logging.info("Training set ==> F1: {}, EM: {}, for {} samples".
                             format(tf1 / train_len, tem / train_len, train_len))

        f1 = 0.
        em = 0.
        val_len = sample[1]

        val_context = dataset['val_context'][:val_len]
        val_question = dataset['val_question'][:val_len]
        val_answer = raw_answers['raw_val_answer'][:val_len]

        val_a_s = np.array([], dtype=np.int32)
        val_a_e = np.array([], dtype=np.int32)

        for i in tqdm(range(val_len // input_batch_size), desc='val set'):
            val_as, val_ae = self.answer(self.model, sess,
                                         np.array(val_context[i * input_batch_size:(i + 1) * input_batch_size]),
                                         np.array(val_question[i * input_batch_size:(i + 1) * input_batch_size]))

            val_a_s = np.concatenate((val_a_s, val_as), axis=0)
            val_a_e = np.concatenate((val_a_e, val_ae), axis=0)

        # a_s and a_e -> (sample_num)
        for i in range(val_len):
            prediction_ids = val_context[i][0][val_a_s[i]:val_a_e[i]+1]
            prediction_answer = ' '.join(rev_vocab[prediction_ids])
            raw_answer = val_answer[i]
            f1 += f1_score(prediction_answer, raw_answer)
            em += exact_match_score(prediction_answer, raw_answer)
            # if i < 10:
            #     print("predict_answer: ", prediction_answer)
            #     print("ground truth: ", raw_answer)
            #     print ("f1: ", f1_score(prediction_answer, raw_answer))

        if log:
            logging.info("val set ==> F1: {}, EM: {}, for {} samples".
                         format(f1 / val_len, em / val_len, val_len))

        if training:
            return tf1/train_len, tem/train_len, f1/val_len, em/val_len
        else:
            return f1/val_len, em/val_len

    def answer(self, model, session, context, question):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}
        input_feed[model.context] = context[:, 0, :]
        input_feed[model.context_m] = context[:, 1, :]
        input_feed[model.question] = question[:, 0, :]
        input_feed[model.question_m] = question[:, 1, :]

        output_feed = [model.s_score, model.e_score]

        s_score, e_score = session.run(output_feed, input_feed)

        a_s = np.argmax(s_score, axis=1)
        a_e = np.argmax(e_score, axis=1)

        return a_s, a_e

    def load_input_feed(self, context, question, answer):
        model = self.model
        input_feed = {}
        input_feed[model.context] = context[:, 0, :]
        input_feed[model.context_m] = context[:, 1, :]
        input_feed[model.question] = question[:, 0, :]
        input_feed[model.question_m] = question[:, 1, :]
        input_feed[model.answer_s] = answer[:, 0]
        input_feed[model.answer_e] = answer[:, 1]
        input_feed[model.starter_learning_rate] = self.start_lr
        return input_feed
