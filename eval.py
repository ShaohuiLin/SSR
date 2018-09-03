#coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import os
import numpy as np
import pdb
from tensorflow.python.keras import backend as K
import tensorflow.contrib.slim as slim

from datasets import dataset_factory
from keras_nets import nets_factory
from preprocessing import preprocessing_factory

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', 'tmp/lenet/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_bool(
    'from_slim', False, 'the model whether from slim')

tf.app.flags.DEFINE_string(
    'eval_dir', 'tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'mnist', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './data/mnist', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'lenet_aulm', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'lenet', 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 28, 'Eval image size')


FLAGS = tf.app.flags.FLAGS


def make_keymap(checkpoint_path):
    checkpoint_path = os.path.join(checkpoint_path)
    reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    keymap = dict()
    for key in var_to_shape_map:
        if key == "global_step" or "mean_rgb" in key or "AuxLogits" in key or "Momentum" in key:
            continue

        newkey = key
        if "resnet" in key and "weights" in key and ("shortcut" in key or "conv" in key):
           newkey = key.replace("weights", "conv2d/kernel")
        elif "Inception" in key and "weight" in key and "Logits" not in key:
            newkey = key.replace("weights", "conv2d/kernel")
        elif "weights" in key:
            newkey = key.replace("weights", "kernel")
        elif "biases" in key:
            newkey = key.replace("biases", "bias")
        model_name = key.split("/")[0]
        if FLAGS.model_name != model_name:
            newkey = newkey.replace(model_name, FLAGS.model_name)

        keymap[key] = newkey

    return keymap

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default() as g:
        K.set_learning_phase(False)
        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        predicts = network_fn(images)

        if FLAGS.max_num_batches:
          num_batches = FLAGS.max_num_batches
        else:
          num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
          checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(tf.nn.softmax(predicts, 1), labels, 1), tf.float32))
            top5_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(tf.nn.softmax(predicts, 1), labels, 5), tf.float32))

        all_acc = 0
        all_top5_acc = 0

        if FLAGS.from_slim:
            keymap = make_keymap(FLAGS.checkpoint_path)
            tf.train.init_from_checkpoint(FLAGS.checkpoint_path, keymap)
            sess.run(tf.global_variables_initializer())
        else:
            restorer = tf.train.Saver()
            restorer.restore(sess, FLAGS.checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(int(num_batches)):
            acc, top5_acc= sess.run([accuracy, top5_accuracy])

            all_acc += acc
            all_top5_acc += top5_acc

            if i % 100 == 0:
                print("step[%d/%d]: accuracy: %g, top5-accuracy: %g" % (i, num_batches, all_acc / (i + 1), all_top5_acc / (i + 1)))

        print("accuracy: %g" % (all_acc / (num_batches)))
        print("top5-accuracy: %g" % (all_top5_acc / (num_batches)))

        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
  tf.app.run()
