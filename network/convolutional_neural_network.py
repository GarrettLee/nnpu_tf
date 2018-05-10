import tensorflow as tf
import network.network_base as network_base
import network.pu_network as pu_network
import helper.cfg_helper as cfg_helper

__author__ = 'garrett_local'


class ConvolutionalNeuralNetwork(pu_network.PuNetwork):

    def __init__(self, cfg, prior=None, trainable=True, do_summarizing=False):
        super(ConvolutionalNeuralNetwork, self).__init__(cfg, prior, trainable)
        if 'weight_decay' in cfg:
            self._weight_decay = float(cfg['weight_decay'])
        else:
            self._weight_decay = 0.005
        self._do_summarizing = do_summarizing

    @network_base.with_graph
    def setup(self):
        with tf.variable_scope('input'):
            super(ConvolutionalNeuralNetwork, self).setup()
            self._x = tf.placeholder('float', shape=(None, 32, 32, 3), name='x')
            self._y = tf.placeholder('float', shape=(None, 1), name='y')
        self._layers.update({
            'x': self._x,
            'y': self._y
        })

        def conv(inputs, output_channel, name='conv', stride=1,
                 padding='SAME', kernel_size=3):
            return self.conv2d((inputs,),
                               (kernel_size, kernel_size),
                               output_channel,
                               (stride, stride),
                               name=name,
                               activation='relu',
                               batch_norm=cfg_helper.to_bool(self._cfg['batch_norm']),
                               padding=padding,
                               weight_decay=self._weight_decay)

        def fc(x, channel_out, name='fc', activation='relu'):
            return self.fc((x, ), channel_out, batch_norm=False, name=name,
                           no_bias=False, activation=activation,
                           weight_decay=self._weight_decay)

        l1 = conv(self._x, 96, name='l1')
        l2 = conv(l1, 96, name='l2')
        l3 = conv(l2, 96, name='l3', stride=2)
        l4 = conv(l3, 192, name='l4')
        l5 = conv(l4, 192, name='l5')
        l6 = conv(l5, 192, name='l6', stride=2)
        l7 = conv(l6, 192, name='l7')
        l8 = conv(l7, 192, name='l8', kernel_size=1, padding='VALID')
        l9 = conv(l8, 10, name='l9', kernel_size=1, padding='VALID')
        l10 = fc(l9, 1000, name='l10')
        l11 = fc(l10, 1000, name='l11')
        l12 = fc(l11, 1, name='l12', activation=None)

        self.sign((l12,), name='prediction')

        if self._risk_type == 'nnPU':
            loss_for_logging, loss_for_update = \
                self.nnpu_loss(
                    (l12, self._y),
                    surrogate_loss='sigmoid',
                    name='nnPU_loss'
                )
            self._layers['loss_for_logging'] = loss_for_logging
            self._layers['loss'] = loss_for_update + self._get_regularization_loss()

        elif self._risk_type == 'uPU':
            loss_for_logging, loss_for_update = self.upu_loss(
                (l12, self._y),
                surrogate_loss='sigmoid',
                name='uPU_loss'
            )
            self._layers['loss'] = loss_for_update + self._get_regularization_loss()
            self._layers['loss_for_logging'] = loss_for_logging
        elif self._risk_type == 'PN':
            log, loss = self.pn_loss(
                (l12, self._y),
                surrogate_loss='sigmoid',
                name='PN_loss'
            )
            self._layers['loss'] = loss + self._get_regularization_loss()
            self._layers['loss_for_logging'] = log

        if self._do_summarizing:
            self.add_summary(self._layers['loss_for_logging'],
                             'training mini batch loss')
