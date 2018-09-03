#coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from keras_nets import lenet_aulm_1
from keras_nets import lenet_aulm_1_ft
from keras_nets import lenet_aulm_2
from keras_nets import lenet_aulm_2_ft
from keras_nets import lenet_aulm_3
from keras_nets import lenet_aulm_3_ft

networks_map = {"lenet_aulm_1": lenet_aulm_1.net,
				"lenet_aulm_1_ft": lenet_aulm_1_ft.net,
				"lenet_aulm_2": lenet_aulm_2.net,
				"lenet_aulm_2_ft": lenet_aulm_2_ft.net,
				"lenet_aulm_3": lenet_aulm_3.net,
				"lenet_aulm_3_ft": lenet_aulm_3_ft.net,}

def get_network_fn(name, num_classes, weight_decay=None, is_training=False, learning_rate=None):
  #if name not in networks_map:
  #  raise ValueError('Name of network unknown %s' % name)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images, **kwargs):
    #arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    with tf.variable_scope(name):
        if learning_rate is None:
            return func(images, num_classes, weight_decay=weight_decay, is_training=is_training)
        else:
            return func(images, num_classes, weight_decay=weight_decay, is_training=is_training, learning_rate=learning_rate)

  return network_fn
