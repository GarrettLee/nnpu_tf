import tensorflow as tf
import network.network_base as network_base
import network.pu_network as pu_network

__author__ = 'garrett_local'


class MultilayerPerceptron(pu_network.PuNetwork):

    def __init__(self, cfg, prior=None, trainable=True, do_summarizing=False):
        super(MultilayerPerceptron, self).__init__(cfg, prior, trainable)
        self._do_summarizing = do_summarizing
        if 'weight_decay' in cfg:
            self._weight_decay = float(cfg['weight_decay'])
        else:
            self._weight_decay = 0.005

    @network_base.with_graph
    def setup(self):
        with tf.variable_scope('input'):
            super(MultilayerPerceptron, self).setup()
            self._x = tf.placeholder('float', shape=(None, 28, 28, 1), name='x')
            self._y = tf.placeholder('float', shape=[None, 1], name='y')
        self._layers['x'] = self._x
        self._layers['y'] = self._y

        x_reshaped = tf.reshape(self._x, shape=(-1, 784), name='x_reshaped')

        def fc(x, channel_out, name='fc'):
            return self.fc((x, ), channel_out, batch_norm=True, name=name,
                           weight_decay=self._weight_decay)

        l1 = fc(x_reshaped, 300, name='layer_1')
        l2 = fc(l1, 300, name='layer_2')
        l3 = fc(l2, 300, name='layer_3')
        l4 = fc(l3, 300, name='layer_4')
        l5 = self.fc((l4,), 1, activation=None, name='layer_5', no_bias=False,
                     weight_decay=self._weight_decay)

        self.sign((l5,), name='prediction')

        if self._risk_type == 'nnPU':
            loss_for_logging, loss_for_update = \
                self.nnpu_loss(
                    (l5, self._y),
                    surrogate_loss='sigmoid',
                    name='nnPU_loss')
            self._layers['loss_for_logging'] = loss_for_logging
            self._layers['loss'] = \
                loss_for_update + self._get_regularization_loss()

        elif self._risk_type == 'uPU':
            loss_for_logging, loss_for_update = self.upu_loss(
                (l5, self._y),
                surrogate_loss='sigmoid',
                name='uPU_loss'
            )
            self._layers['loss'] = loss_for_update + self._get_regularization_loss()
            self._layers['loss_for_logging'] = loss_for_logging

        elif self._risk_type == 'PN':
            log, loss = self.pn_loss(
                (l5, self._y),
                surrogate_loss='sigmoid',
                name='PN_loss'
            )
            self._layers['loss'] = loss + self._get_regularization_loss()
            self._layers['loss_for_logging'] = log
        if self._do_summarizing:
            self.add_summary(self._layers['loss_for_logging'],
                             'training mini batch loss')
