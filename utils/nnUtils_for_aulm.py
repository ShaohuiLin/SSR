# coding: utf-8
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import layers
from tensorflow.python.keras._impl.keras import activations
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import initializers
from tensorflow.python.keras._impl.keras import regularizers
from tensorflow.python.keras._impl.keras import constraints

class AULMConv2d(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 p=1.0,
                 Lambda=0.5,
                 name="conv2d",
                 is_training=True,
                 **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.p = p
        self.Lambda = Lambda
        self.is_training = is_training
        self.layername = name
        super(AULMConv2d, self).__init__(name=self.layername, **kwargs)

    def build(self, input_shape):
        with tf.variable_scope(self.layername):
            self.kernel = self.add_weight(name='kernel', shape=[self.kernel_size[0], self.kernel_size[1],
                                                                input_shape[3], self.filters],
                                          initializer=tf.contrib.layers.xavier_initializer_conv2d(), trainable=True)

            self.mask = self.add_weight(name='mask', shape=[self.filters],
                                        initializer=tf.initializers.ones(), trainable=False)
            if self.use_bias:
                self.bias = self.add_weight(name='bias', shape=[self.filters], initializer=tf.initializers.zeros(),
                                            trainable=True)

            if self.is_training:
                if self.use_bias:
                    self.F = self.add_weight(name='F',
                                             shape=[self.filters,
                                                        input_shape[3] * self.kernel_size[0] * self.kernel_size[1] + 1],
                                             initializer=tf.initializers.ones(),
                                             trainable=False)

                    self.Y = self.add_weight(name='Y',
                                             shape=[self.filters,
                                                        input_shape[3] * self.kernel_size[0] * self.kernel_size[1] + 1],
                                             initializer=tf.initializers.zeros(), trainable=False)

                    mkernel = tf.concat([tf.reshape(tf.transpose(self.kernel, [3, 2, 0, 1]), [self.filters, -1]),
                                         tf.reshape(self.bias, [self.filters, 1])], 1, name="mkernel")

                else:
                    self.F = self.add_weight(name='F',
                                             shape=[self.filters,
                                                    input_shape[3] * self.kernel_size[0] * self.kernel_size[1]],
                                             initializer=tf.initializers.ones(),
                                             trainable=False)

                    self.Y = self.add_weight(name='Y',
                                             shape=[self.filters,
                                                    input_shape[3] * self.kernel_size[0] * self.kernel_size[1]],
                                             initializer=tf.initializers.zeros(), trainable=False)

                    mkernel = tf.reshape(tf.transpose(self.kernel, [3, 2, 0, 1]), [self.filters, -1], name="mkernel")

                T2 = mkernel + 1 / self.p * self.Y
                F_update = tf.assign(self.F, T2 * tf.reshape(tf.maximum(
                    tf.norm(T2, ord=2, axis=1) - self.Lambda / self.p, 0) / (tf.norm(T2, ord=2, axis=1) + 1e-9), [-1, 1]), name="F_update")
                with tf.control_dependencies([F_update]):
                    Y_update = tf.assign(self.Y, self.Y + self.p * (mkernel - self.F), name="Y_update")
                tf.get_default_graph().add_to_collection("UPDATE_FY", Y_update)

                # T1 = tf.stop_gradient(F_update - 1 / self.p * Y_update)
                T1 = self.F - 1 / self.p * self.Y
                tf.losses.add_loss(self.p/2 * tf.pow(tf.norm(mkernel - T1, ord='fro', axis=[0, 1]), 2))
                # tf.losses.mean_squared_error(labels=T1, predictions=mkernel, weights=self.p/2)

                mask = tf.assign(self.mask, 1 - tf.to_float(tf.equal(tf.reduce_sum(self.F, 1), 0)))
                w_compress = 1 - tf.reduce_sum(mask) / self.filters
                tf.get_default_graph().add_to_collection("COMPRESS", w_compress)

        super(AULMConv2d, self).build(input_shape)

    def call(self, inputs, **kwargs):
        out = tf.nn.conv2d(inputs, self.kernel, strides=[1, self.strides[0], self.strides[1], 1],
                           padding=self.padding)
        if self.use_bias:
            out = out + self.bias
        if self.activation is not None:
            out = activations.get(self.activation)(out)

        return out

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(activations.get(self.activation)),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(initializers.get(self.kernel_initializer)),
            'bias_initializer': initializers.serialize(initializers.get(self.bias_initializer)),
            'kernel_regularizer': regularizers.serialize(regularizers.get(self.kernel_regularizer)),
            'bias_regularizer': regularizers.serialize(regularizers.get(self.bias_regularizer)),
            'activity_regularizer': regularizers.serialize(regularizers.get(self.activity_regularizer)),
            'kernel_constraint': constraints.serialize(constraints.get(self.kernel_constraint)),
            'bias_constraint': constraints.serialize(constraints.get(self.bias_constraint))
        }
        base_config = {"name": self.layername}
        return dict(list(base_config.items()) + list(config.items()))


class MaskConv2d(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name="conv2d",
                 is_training=True,
                 **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.is_training = is_training
        self.layername = name
        super(MaskConv2d, self).__init__(name=self.layername, **kwargs)

    def build(self, input_shape):
        with tf.variable_scope(self.layername):
            self.kernel = self.add_weight(name='kernel', shape=[self.kernel_size[0], self.kernel_size[1],
                                                                input_shape[3], self.filters],
                                          initializer=tf.contrib.layers.xavier_initializer_conv2d(), trainable=self.is_training)

            self.mask = self.add_weight(name='mask', shape=[self.filters],
                                        initializer=tf.initializers.ones(), trainable=False)
            if self.use_bias:
                self.bias = self.add_weight(name='bias', shape=[self.filters], initializer=tf.initializers.zeros(),
                                            trainable=self.is_training)

        super(MaskConv2d, self).build(input_shape)

    def call(self, inputs, **kwargs):
        out = tf.nn.conv2d(inputs, self.kernel * self.mask, strides=[1, self.strides[0], self.strides[1], 1],
                           padding=self.padding)
        if self.use_bias:
            out = out + self.bias * self.mask
        if self.activation is not None:
            out = activations.get(self.activation)(out)

        return out

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(activations.get(self.activation)),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(initializers.get(self.kernel_initializer)),
            'bias_initializer': initializers.serialize(initializers.get(self.bias_initializer)),
            'kernel_regularizer': regularizers.serialize(regularizers.get(self.kernel_regularizer)),
            'bias_regularizer': regularizers.serialize(regularizers.get(self.bias_regularizer)),
            'activity_regularizer': regularizers.serialize(regularizers.get(self.activity_regularizer)),
            'kernel_constraint': constraints.serialize(constraints.get(self.kernel_constraint)),
            'bias_constraint': constraints.serialize(constraints.get(self.bias_constraint))
        }
        base_config = {"name": self.layername}
        return dict(list(base_config.items()) + list(config.items()))