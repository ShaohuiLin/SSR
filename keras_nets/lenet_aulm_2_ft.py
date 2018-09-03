#c# coding:utf-8

import tensorflow as tf
from tensorflow.python.keras.layers import *
from utils.nnUtils_for_aulm import *

def net(inputs, num_classes, weight_decay=None, is_training=False):

    x = MaskConv2d(20, (5, 5), activation='relu', padding='valid', name='conv1', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    x = MaskConv2d(50, (5, 5), activation='relu', padding='valid', name='conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(500, (4, 4), activation="relu", padding="valid", name='fc1', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Conv2D(num_classes, (1, 1), padding="valid", name='fc2')(x)
    y = Flatten(name='fc2/squeezed')(x)

    return y