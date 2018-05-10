import os
import re
import numpy as np
import tensorflow as tf

import helper.file_helper as file_manager
import network.network_base as network_base

__author__ = 'garrett_local'


class TrainerBase(object):
    def __init__(self, cfg, do_summarizing=False, summary_path=None):
        self._cfg = cfg
        self._session = None
        self._network = None
        self._do_summarizing = do_summarizing
        self._summary_writer = None
        self._summaries = None
        self._iter = 0
        self._train_op = None
        self._summary_path = summary_path
        if self._do_summarizing:
            if self._summary_path is None:
                raise ValueError(
                    'summary_path must be set if do_summarizing is ture.'
                )

    def __del__(self):
        if self._session is not None:
            self._session.close()

    @property
    def iter(self):
        return self._iter

    def get_train_op(self):
        assert self._network is not None, 'Network not set.'
        if self._cfg['algorithm'] == 'AdaGrad':
            if 'accumulator' in self._cfg:
                accum = float(self._cfg['accumulator'])
            else:
                accum = 0.1
            optimizer = tf.train.AdagradOptimizer(
                float(self._cfg['learning_rate']),
                initial_accumulator_value=accum
            )
        elif self._cfg['algorithm'] == 'Adam':
            optimizer = tf.train.AdamOptimizer(
                float(self._cfg['learning_rate'])
            )
        else:
            raise NotImplementedError
        return optimizer.apply_gradients(
            self._network.gradient(self._network.get_output('loss'),
                                   do_summarizing=self._do_summarizing)
        )

    def setup_network(self, network):
        assert issubclass(network.__class__, network_base.NetworkBase), \
            'network must be a subclass of NetworkBase class.'
        self._network = network
        graph = self._network.graph
        with graph.as_default():
            network.setup()
            self._train_op = self.get_train_op()
            self._summaries = network.get_summaries()
            init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._session = tf.Session(graph=graph, config=config)
        if self._do_summarizing:
            self._summary_writer = tf.summary.FileWriter(
                self._summary_path,
                self._session.graph
            )
        self._session.run(init)

    def train(self, fed_data):
        """
        fed_data must have the same format as the
        arguments of function generate_feed_dict_for_training implemented in
        pu_network.py.
        :param fed_data: a list of ndarray. mini batch.
        :return: a float. loss.
        """
        feed_dict, _ = self._network.generate_feed_dict_for_training(fed_data)
        fetch_list = [self._train_op,
                      self._network.get_output('loss_for_logging')]
        if self._do_summarizing:
            summaries_pos = len(fetch_list)
            fetch_list.append(self._summaries)
        vals = self._session.run(fetch_list, feed_dict=feed_dict)
        self._iter += 1
        loss = vals[1]
        if self._do_summarizing:
            self._summary_writer.add_summary(vals[summaries_pos], self._iter)
            self._summary_writer.flush()
        return loss

    def evaluate_loss(self, fed_data_iter):
        losses_sum = 0
        sample_num = 0
        for fed_data in fed_data_iter:
            feed_dict, batch_size = \
                self._network.generate_feed_dict_for_testing(fed_data[:2])
            loss = self._session.run(
                self._network.get_output('loss_for_logging'),
                feed_dict=feed_dict
            )
            sample_num += batch_size
            losses_sum += loss * batch_size
        return float(losses_sum) / float(sample_num)

    def test(self, fed_data):
        """
        fed_data must have the same format as the arguments of function
        generate_feed_dict_for_testing implemented in pu_network.py.
        :param fed_data: a list of ndarray. mini batch.
        :return: an ndarry. the prediction.
        """
        return self._session.run(
            self._network.get_output('prediction'),
            feed_dict=self._network.generate_feed_dict_for_testing(fed_data)[0]
        )

    def evaluate_error(self, fed_data_iter):
        error_sum = 0
        sample_num = 0
        for fed_data in fed_data_iter:
            feed_dict, batch_size = \
                self._network.generate_feed_dict_for_testing(fed_data[:1])
            prediction = self._session.run(
                self._network.get_output('prediction'),
                feed_dict=feed_dict
            )
            labels = fed_data[1]
            error_sum += np.sum(prediction != labels)
            sample_num += batch_size
        return float(error_sum) / float(sample_num)

    def save_model(self, path, save_type='tensorflow_save'):
        assert self._session is not None, ('Execute function self.setup_network '
                                           'before saving.')
        file_manager.create_if_not_exist(path)
        if save_type == 'tensorflow_save':
            with self._session.graph.as_default():
                saver = tf.train.Saver()
                saver.save(
                    self._session,
                    os.path.join(path, 'model.ckpt'),
                    self._iter
                )
        if save_type == 'npy_save':
            self._network.save_network_to_npy(
                os.path.join(path,
                             'model_{0}.npy'.format(self._iter)),
                self._session
            )

    def load_model(self, path, iter_ind=None, load_type='tensorflow_save'):
        if iter_ind is None:
            if load_type == 'tensorflow_save':
                last_modified_model_name = file_manager.last_modified(
                    os.path.join(path,
                                 'model.ckpt-*.data*')
                )
                if last_modified_model_name is None:  # path doesn't exist.
                    return None
                iter_ind = re.findall(
                    '\d+',
                    os.path.basename(last_modified_model_name)
                )[0]
                with self._session.graph.as_default():
                    saver = tf.train.Saver()
                    try:
                        saver.restore(
                            self._session,
                            os.path.join(path,
                                         'model.ckpt-{0}'.format(iter_ind))
                        )
                    except (tf.errors.NotFoundError,
                            tf.errors.InvalidArgumentError):  # File doesn't exist.
                        return None

            elif load_type == 'npy_save':
                last_modified_model_name = file_manager.last_modified(
                    os.path.join(path,
                                 'model_*.npy')
                )
                if last_modified_model_name is None:  # path doesn't exist.
                    return None
                self._network.restore_network_from_npy(
                    last_modified_model_name,
                    self._session
                )
                iter_ind = re.findall(
                    '\d+',
                    os.path.basename(last_modified_model_name)
                )[0]
            else:
                raise ValueError('load_type is illegal.')
        else:
            if load_type == 'tensorflow_save':
                with self._session.graph.as_default():
                    saver = tf.train.Saver()
                    try:
                        saver.restore(
                            self._session,
                            os.path.join(
                                path,
                                'model.ckpt-{0}'.format(int(iter_ind))
                            )
                        )
                    except (tf.errors.NotFoundError,
                            tf.errors.InvalidArgumentError):  # File doesn't exist.
                        return None
            elif load_type == 'npy_save':
                try:
                    self._network.restore_network_from_npy(
                        os.path.join(
                            path,
                            'model_{0}.npy'.format(int(iter_ind))
                        ),
                        self._session
                    )
                except IOError:  # File doesn't exist.
                    return None
            else:
                raise ValueError('load_type is illegal.')
        if iter_ind is not None:
            self._iter = iter_ind
            return iter_ind
        else:
            return None
