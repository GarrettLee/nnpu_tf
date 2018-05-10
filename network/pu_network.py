import tensorflow as tf
import network.network_base as network_base

__author__ = 'garrett_local'


class PuNetwork(network_base.NetworkBase):

    def __init__(self, cfg, prior, trainable=True):
        super(PuNetwork, self).__init__(trainable=trainable)
        assert 'risk_type' in cfg
        assert cfg['risk_type'] in ['nnPU', 'uPU', 'PN']
        self._cfg = cfg
        self._risk_type = cfg['risk_type']
        self._prior_val = prior
        self._positive = 1
        self._negative = -1
        self._x = None
        self._y = None
        self.__beta = None
        self.__gamma = None
        self.__prior = None
        self.__test = True

    @property
    def _prior(self):
        if self.__prior is None:
            self.__prior = tf.placeholder('float', shape=(), name='prior')
        return self.__prior

    @property
    def _gamma(self):
        if self.__gamma is None:
            self.__gamma = tf.placeholder('float', shape=(), name='gamma')
        return self.__gamma

    @property
    def _beta(self):
        if self.__beta is None:
            self.__beta = tf.placeholder('float', shape=(), name='beta')
        return self.__beta

    @property
    def risk_type(self):
        return self._risk_type

    def _parse_loss_function(self, surrogate_loss):
        assert surrogate_loss in ['sigmoid', 'logistics', 'zero-one']
        assert self._negative == -1 and self._positive == 1
        if surrogate_loss == 'sigmoid':
            return lambda network_out, y, name: \
                tf.nn.sigmoid(-network_out * y, name=name)
        elif surrogate_loss == 'logistics':
            return lambda network_out, y, name:\
                tf.nn.softplus(-network_out * y, name=name)
        elif surrogate_loss == 'zero-one':
            return lambda network_out, y, name:\
                (tf.cast(tf.greater(-network_out * y, 0.), tf.float32))
        else:
            raise NotImplementedError('Unknown loss function: {}'
                                      .format(surrogate_loss))

    @network_base.with_graph
    def _calculate_losses(self, network_out, labels, prior, surrogate_loss, logging=False):
        assert network_out.shape.ndims == 2
        assert network_out.shape[1] == 1
        if not logging:
            loss_func = self._parse_loss_function(surrogate_loss)
        else:
            loss_func = self._parse_loss_function('zero-one')
        positive = tf.cast(tf.equal(labels, self._positive), tf.float32, name='positive_label')
        unlabeled = tf.cast(tf.equal(labels, self._negative), tf.float32, name='negative_label')
        num_positive = tf.maximum(1., tf.reduce_sum(positive), name='positive_number')
        num_unlabeled = tf.maximum(1., tf.reduce_sum(unlabeled), name='unlabeled_number')
        losses_positive = loss_func(network_out, self._positive, 'positive_loss')
        losses_negative = loss_func(network_out, self._negative, 'negative_loss')
        positive_risk = tf.reduce_sum(prior * positive / num_positive *
                                      losses_positive, name='positive_risk')
        negative_risk = tf.reduce_sum((unlabeled / num_unlabeled - prior *
                                       positive / num_positive) *
                                      losses_negative, name='negative_risk')
        return positive_risk, negative_risk

    @network_base.layer
    def nnpu_loss(self, inputs, surrogate_loss, name='nnpu_loss'):
        assert len(inputs) == 2
        network_out = inputs[0]
        labels = inputs[1]
        positive_risk, negative_risk = self._calculate_losses(network_out,
                                                              labels,
                                                              self._prior,
                                                              surrogate_loss)
        positive_log, negative_log = self._calculate_losses(network_out,
                                                            labels,
                                                            self._prior,
                                                            surrogate_loss,
                                                            logging=True)
        is_ga = tf.less(negative_risk, -self._beta)
        loss_for_logging = tf.cond(tf.less(negative_log, -self._beta),
                                   lambda: positive_log - self._beta,
                                   lambda: positive_log + negative_log)
        loss_for_update = tf.cond(is_ga,
                                  lambda: -self._gamma * negative_risk,
                                  lambda: positive_risk + negative_risk)
        return loss_for_logging, loss_for_update

    @network_base.layer
    def upu_loss(self, inputs, surrogate_loss, name='upu_loss'):
        assert len(inputs) == 2
        network_out = inputs[0]
        labels = inputs[1]
        positive_risk, negative_risk = self._calculate_losses(network_out,
                                                              labels,
                                                              self._prior,
                                                              surrogate_loss)
        positive_log, negative_log = self._calculate_losses(network_out,
                                                            labels,
                                                            self._prior,
                                                            surrogate_loss,
                                                            logging=True)
        upu_loss = positive_risk + negative_risk
        upu_log = positive_log + negative_log
        return upu_log, upu_loss

    @network_base.with_graph
    def _cal_pn_loss(self, network_out, labels, surrogate_loss):
        positive = tf.cast(tf.equal(labels, self._positive), tf.float32,
                           name='positive_label')
        negative = tf.cast(tf.equal(labels, self._negative), tf.float32,
                           name='negative_label')

        num_positive = tf.maximum(1., tf.reduce_sum(positive), name='positive_number')
        num_negative = tf.maximum(1., tf.reduce_sum(negative), name='negative_number')

        loss_func = self._parse_loss_function(surrogate_loss)
        positive_losses = loss_func(network_out, self._positive, 'positive_losses')
        negative_losses = loss_func(network_out, self._negative, 'negative_losses')
        losses = tf.reduce_sum(self._prior * positive / num_positive *
                               positive_losses + (1 - self._prior) * negative /
                               num_negative * negative_losses, name='pu_risk')
        if self.__test:
            if not hasattr(self, 'cal_pn_loss_testing'):
                self.cal_pn_loss_testing = {
                    'negative': [negative],
                    'positive': [positive],
                    'num_positive': [num_positive],
                    'num_negative': [num_negative],
                }
            else:
                self.cal_pn_loss_testing['negative'].append(negative)
                self.cal_pn_loss_testing['positive'].append(positive)
                self.cal_pn_loss_testing['num_positive'].append(num_positive)
                self.cal_pn_loss_testing['num_negative'].append(num_negative)
        return tf.reduce_mean(losses)

    @network_base.layer
    def pn_loss(self, inputs, surrogate_loss, name='pn_loss'):
        assert len(inputs) == 2
        network_out = inputs[0]
        labels = inputs[1]

        # loss for update.
        with tf.name_scope('{}_loss'.format(name)):
            loss = self._cal_pn_loss(network_out, labels, surrogate_loss)

        # loss for logging.
        with tf.name_scope('{}_log'.format(name)):
            log = self._cal_pn_loss(network_out, labels, 'zero-one')
        return log, loss

    @network_base.with_graph
    def generate_feed_dict_for_training(self, fed_data):
        feed_dict = super(PuNetwork, self).\
            generate_feed_dict_for_training([])
        assert len(fed_data) == 2
        x, y = fed_data
        if self._risk_type == 'PN':
            feed_dict.update({
                self._x: x,
                self._y: y,
                self._prior: self._prior_val
            })
        else:
            if self._risk_type == 'nnPU':
                feed_dict.update({
                    self._x: x,
                    self._y: y,
                    self._beta: float(self._cfg['beta']),
                    self._gamma: float(self._cfg['gamma']),
                    self._prior: self._prior_val
                })
            if self._risk_type == 'uPU':
                feed_dict.update({
                    self._x: x,
                    self._y: y,
                    self._prior: self._prior_val
                })
        return feed_dict, x.shape[0]

    @network_base.with_graph
    def generate_feed_dict_for_testing(self, fed_data):
        feed_dict = super(PuNetwork, self).\
            generate_feed_dict_for_testing([])
        assert len(fed_data) == 1 or len(fed_data) == 2
        if len(fed_data) == 1:
            x, = fed_data
            feed_dict.update({
                self._x: x
            })
        else:
            x, y = fed_data
            if self._risk_type == 'PN':
                feed_dict.update({
                    self._x: x,
                    self._y: y,
                    self._prior: self._prior_val
                })
            elif self._risk_type == 'nnPU':
                feed_dict.update({
                    self._x: x,
                    self._y: y,
                    self._beta: float(self._cfg['beta']),
                    self._gamma: float(self._cfg['gamma']),
                    self._prior: self._prior_val
                })
            elif self._risk_type == 'uPU':
                feed_dict.update({
                    self._x: x,
                    self._y: y,
                    self._prior: self._prior_val
                })
            else:
                assert False
        return feed_dict, x.shape[0]
