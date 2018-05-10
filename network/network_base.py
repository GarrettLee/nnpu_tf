import abc
import tensorflow as tf
import numpy as np

__author__ = 'garrett_local'


def layer(op):
    """
    A decorator for network op.

    Via this decorator:
    1. Output of op will be stored automatically.
    2. TensorBoard summaries will be performed automatically.
    :param op: network op.
    :return: decorated op.
    """

    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name',
                                 self._get_unique_name(op.__name__.strip('_')))
        if 'do_summarizing' in kwargs:
            do_summarizing = kwargs.pop('do_summarizing')
        else:
            do_summarizing = False

        with tf.variable_scope(name):
            layer_output = op(self, *args, **kwargs)
        self._layers[name] = layer_output
        if do_summarizing:
            variables = self.trainable_variables(name)
            if len(variables) != 0:
                summaries = []
                for variable in variables:
                    summaries.append(tf.summary.histogram("{0}/hist".
                                                          format(variable.name),
                                                          variable))
                summaries = tf.summary.merge(summaries)
                self._variable_summaries[name] = summaries
        return layer_output
    return layer_decorated


def with_graph(func):
    def func_with_graph(self, *args, **kwargs):
        default_graph = tf.get_default_graph()
        if default_graph is not self.graph:
            with self.graph.as_default():
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)
    return func_with_graph


class NetworkBase(object):
    def __init__(self, trainable=True):
        self.graph = tf.Graph()
        self._layers = {}
        self._variable_summaries = {}
        self._gradient_summaries = {}
        self._trainable = trainable
        self._is_training = None

    @with_graph
    def setup(self):
        self._layers['is_training'] = tf.placeholder(tf.bool,
                                                     shape=[],
                                                     name='is_training')

    def add_summary(self, tensor, name):
        ndims = tensor.shape.ndims
        if ndims == 0:
            self._variable_summaries[name] = tf.summary.scalar(name, tensor)
        else:
            self._variable_summaries[name] = tf.summary.histogram('{0}/hist'.
                                                                  format(name),
                                                                  tensor)

    def get_output(self, layer_name):
        """
        Get outputs of a specified layer.
        :param layer_name: a basestring which is one of the keys of self._layers.
        :return: a TensorFlow tensor.
        """
        try:
            l = self._layers[layer_name]
        except KeyError:
            print(self._layers.keys())
            raise KeyError('Unknown layer name to be fetched: {0}'
                           .format(layer_name))
        return l

    def _get_unique_name(self, base):
        """
        Add a id number as prefix to a string, make the string unique.
        """
        id = sum(t.startswith(base + '_') for t, _ in self._layers.items()) + 1
        return '{0}_{1}'.format(base, id)

    @with_graph
    def gradient(self, y, tvars=None, do_summarizing=False):
        """
        Get the gradients of y above multiple variables.
        :param y: a TensorFlow Tensor. Dependent variable.
        :param tvars: a list of TensorFlow Tensor. All must be arguments of y.
        :param do_summarizing: If ture, summaries will be appended into
                    self._gradient_summaries.
        :return a list of Tensors. Each element of the list is a tuple of two
                    Tensor. The first Tensor in the list is gradient and the
                    second Tensor is its corresponding trainable variable.
        """
        if tvars is None:
            tvars = tf.trainable_variables()
        grads = tf.gradients(y, tvars)
        if do_summarizing:
            for idx in range(len(tvars)):
                name = '{0}_{1}'.format(y.name, tvars[idx].name)
                # Prevent from duplicating.
                if name not in self._gradient_summaries:
                    self._gradient_summaries[name] = \
                        tf.summary.histogram('grad/{0}'.format(name),
                                             grads[idx])
        return zip(grads, tvars)

    @with_graph
    def get_summaries(self):
        return tf.summary.merge_all()

    @with_graph
    def save_network_to_npy(self, data_path, session):
        """
        Save network trainable variable to a numpy npy file.

        Created npy file should be restore with function
        self.restore_network_from_npy.
        :param data_path: basestring. Path to save the npy file.
        :param session: TensorFlow Session.
        """
        tvars = self.trainable_variables()
        values = session.run(tvars)
        values_dicts = []
        for idx, value in enumerate(values):
            values_dict = {}
            name = tvars[idx].name.split('/')[-1]
            name_len = len(name)
            scope = tvars[idx].name[:-1 * name_len - 1]
            values_dict['scope'] = scope
            values_dict['name'] = name.split(':')[0]
            values_dict['value'] = value
            values_dicts.append(values_dict)
        np.save(data_path, values_dicts)

    @with_graph
    def restore_network_from_npy(self, data_path, session,
                                 ignore_missing=False):
        """
        Restore network trainable variable from a numpy npy file.

        This npy file must has been created with self.save_network_to_file
        function.
        :param data_path: basestring. Path to npy file.
        :param session: Tensorflow Session.
        :param ignore_missing: if it is set true, variable missing will be
                    ignore.
        """
        data_dicts = np.load(data_path)
        for data_dict in data_dicts:
            with tf.variable_scope(data_dict['scope'], reuse=True):
                try:
                    var = tf.get_variable(data_dict['name'])
                    session.run(var.assign(data_dict['value']))
                    print('assign pretrain model to {0}/{1}'
                          .format(data_dict['scope'], data_dict['name']))
                except ValueError:
                    print('ignore {0}/{1}'.format(data_dict['scope'],
                                                  data_dict['name']))
                    if not ignore_missing:
                        raise

    @with_graph
    def trainable_variables(self, scope=None):
        """
        Get trainable variables in a specified variable scope.
        :param scope: name of a variable scope, can also be None, in which case
                    all trainable variables in the whole graph will be return.
        :return: trainable variables in a specified variable scope. It is a list
                    of dict. each dict corresponds to a trainable variable.
        """
        if scope is None:
            return tf.trainable_variables()  # return all variables.
        else:
            with tf.variable_scope(scope) as scope:
                return scope.trainable_variables()

    @layer
    def fc(self, inputs, channel_out, name='fc',
           activation='relu', weights_initializer=None, trainable=None,
           no_bias=True, batch_norm=False, weight_decay=0.005):
        """
        FC layer.
        :param inputs: a tuple with only one element. The element must be a
                    tensor with shape [BATCH_SIZE, CHANNEL_IN], or a tensor
                    with shape [BATCH_SIZE, HEIGHT, WIDTH, CHANNEL_IN] for
                    output of conv2d layer.
        :param channel_out: output channel number.
        :param name: name of the layer.
        :param activation: a basestring. Name of a implemented activation
                    function.
        :param weights_initializer: a TensorFlow initializer for weights
                    initialization.
        :param trainable: bool.
        :return: a tensor of fc output.
        """
        assert len(inputs) == 1, \
            'fc layer accept only one tensor for input.'

        # reshape output of conv layer.
        input_shape = inputs[0].shape.as_list()
        if len(input_shape) == 4:
            inputs = tf.reshape(
                inputs[0],
                [-1, input_shape[1] * input_shape[2] * input_shape[3]]
            ),
        assert inputs[0].shape.ndims == 2, \
            'Shape of fc input must be [BATCH_SIZE, CHANNEL_IN].'
        if trainable is None:
            trainable = self._trainable
        if weights_initializer is None:

            # There may be a difference between this initializer and the
            # initializer used in the original codes:
            # https://github.com/kiryor/nnPUlearning.git. Truncated normal
            # initializer is used here, while it seems standard normal
            # initializer is used in the original codes.
            weights_initializer = \
                tf.contrib.layers.variance_scaling_initializer(
                    factor=1.0,
                    mode='FAN_IN',
                    uniform=False
                )
        biases_initializer = tf.constant_initializer(0.0)
        channel_in = inputs[0].shape[1]
        with tf.variable_scope('weights'):
            weghts = tf.get_variable(
                'W',
                [channel_in, channel_out],
                initializer=weights_initializer,
                trainable=trainable,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
            )

        fc = tf.matmul(inputs[0], weghts)
        if not no_bias:
            with tf.variable_scope('biases'):
                biases = tf.get_variable('b', [1, channel_out],
                                         initializer=biases_initializer,
                                         trainable=trainable)
            fc = fc + biases
        if batch_norm:
            fc = self.batch_normalization((fc,), (1,))
        if activation is not None:
            activation_op = self._parse_activation(activation)
            fc = activation_op(fc)
        return fc

    @layer
    def conv2d(self, inputs, kernel_shape, output_channel, stride, name,
               no_bias=False, activation=None, padding='SAME', trainable=None,
               weight_decay=0.005, batch_norm=False):
        assert len(inputs) == 1
        assert \
            (len(kernel_shape) == 2)\
            & (len(stride) == 2) \
            & isinstance(output_channel, int)
        if trainable is None:
            trainable = self._trainable
        input_channel = inputs[0].get_shape()[-1]

        def convolve(conv_input, k):
            return tf.nn.conv2d(conv_input, k,
                                [1, stride[0], stride[1], 1],
                                padding=padding)
        weights_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=1,
            mode='FAN_IN',
            uniform=False)
        biases_initializer = tf.constant_initializer(0.0)
        with tf.variable_scope('weights'):
            kernel = tf.get_variable(
                'weights',
                [kernel_shape[0], kernel_shape[1], input_channel, output_channel],
                initializer=weights_initializer,
                trainable=trainable,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        conv = convolve(inputs[0], kernel)
        if not no_bias:
            with tf.variable_scope('biases'):
                biases = tf.get_variable('biases',
                                         [output_channel],
                                         initializer=biases_initializer,
                                         trainable=trainable,
                                         regularizer=None)
            conv = tf.nn.bias_add(conv, biases)
        if batch_norm:
            conv = self.batch_normalization((conv,), (3,))
        if activation is not None:
            activation = self._parse_activation(activation)
            conv = activation(conv)
        return conv

    @layer
    def sigmoid(self, inputs, name='sigmoid'):
        assert len(inputs) == 1
        return tf.nn.sigmoid(inputs[0])

    @layer
    def sign(self, inputs, name='sign', non_zeros=False):
        assert len(inputs) == 1
        if non_zeros:
            return (inputs[0] > 0) * 2. - 1.
        else:
            return tf.sign(inputs[0], name=name)

    @layer
    def dropout(self, inputs, keep_prob, name):
        """
        Dropout layer.
        :param inputs: a tuple of one Tensor.
        :param keep_prob: a scalar tensor.
        :param name: name of the layer.
        :return: dropout output, usually followed by an fc layer.
        """
        assert len(inputs) == 1, \
            'Dropout layer accept only one tensor for input.'
        return tf.nn.dropout(inputs[0], keep_prob, name=name)

    @layer
    def batch_normalization(self, inputs, channel_axes,
                            name='batch_normalization', epsilon=2e-05,
                            decay=0.9):
        assert len(inputs) == 1
        assert inputs[0].shape.ndims >= np.max(channel_axes)
        x = inputs[0]
        is_training = self.get_output('is_training')
        parameters_shape = [x.shape.as_list()[i]
                            for i in range(x.shape.ndims)
                            if i in channel_axes]
        scale = tf.Variable(tf.ones(parameters_shape), name='scale')
        shift = tf.Variable(tf.zeros(parameters_shape), name='shift')

        # For training.
        axis = [i for i in range(x.shape.ndims) if i not in channel_axes]
        mean, variance = tf.nn.moments(x, axis)
        moving_average = tf.train.ExponentialMovingAverage(decay=decay)

        def batch_norm(m, v):
            return tf.nn.batch_normalization(x, m, v, shift, scale,
                                             epsilon)

        def batch_norm_with_update():
            average_op = moving_average.apply([mean, variance])
            with tf.control_dependencies([average_op]):
                # tf.identity makes sure average_op will be evaluated whenever
                # mean_for_training or variance_for_training are evaluated.
                # see: https://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for
                mean_for_training = tf.identity(mean)
                variance_for_training = tf.identity(variance)
            return batch_norm(mean_for_training, variance_for_training)

        def batch_norm_without_update():
            # For inference.
            mean_for_inference = moving_average.average(mean)
            variance_for_inference = moving_average.average(variance)
            return batch_norm(mean_for_inference, variance_for_inference)

        return tf.cond(
            is_training,
            batch_norm_with_update,
            batch_norm_without_update
        )

    @staticmethod
    def _parse_activation(activation):
        """
        Return a TensorFlow activation op according to its name.
        :param activation: basestring. Name of a implemented activation.
        :return: TensorFlow op.
        """
        if activation == 'relu':
            return tf.nn.relu
        if activation == 'tanh':
            return tf.nn.tanh
        if activation == 'sigmoid':
            return tf.nn.sigmoid
        if activation == 'softmax':
            return tf.nn.softmax
        if activation == 'softplus':
            return tf.nn.softplus
        raise Exception('Unknown activation type.')

    @property
    def is_training(self):
        return self._layers['is_training']

    def generate_feed_dict_for_training(self, fed_data):
        assert len(fed_data) == 0
        return {self.is_training: True}

    def generate_feed_dict_for_testing(self, fed_data):
        assert len(fed_data) == 0
        return {self.is_training: False}

    @with_graph
    def _get_regularization_loss(self):
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(regularization_losses) == 0:
            return 0
        loss = tf.add_n(regularization_losses)
        return loss
