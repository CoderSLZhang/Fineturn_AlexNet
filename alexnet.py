# zhangshulin
# zhangslwork@yeah.net
# 2018-1-17


import tensorflow as tf
import os
import numpy as np
from urllib.request import urlretrieve


WEIGHTS_URL = 'http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy'
WEIGHTS_FILE = './alexnet_weights.npy'


class AlexNet:

    def __init__(self, image_shape=(227, 227, 3), classes_num=1000, freeze_layer_indexes=()):
        self._image_shape = image_shape
        self._classes_num = classes_num
        self._freeze_layer_indexes = freeze_layer_indexes
        self._X, self._keep_prob = self._init_placeholders(image_shape)
        self._output_Z, self._output_A = self._create()
        self.last_percent_reported = None


    def load_imagenet_weights(self, session, skip_layer_indexes=()):
        weights_dict = self._get_weights_dict()

        for layer_name in weights_dict:
            layer_index = int(layer_name[-1])

            if layer_index not in skip_layer_indexes:
                weights, bias = weights_dict[layer_name]

                with tf.variable_scope(layer_name, reuse=True):
                    weights_var = tf.get_variable(name='weights')
                    bias_var = tf.get_variable(name='bias')
                    session.run(weights_var.assign(weights))
                    session.run(bias_var.assign(bias))


    def _get_weights_dict(self):
        self._maybe_download(243861814, force=True)
        return np.load(WEIGHTS_FILE, encoding='bytes').item()
    
        
    def _maybe_download(self, expected_bytes, force=False):
        dest_filename = WEIGHTS_FILE
        
        statinfo = os.stat(dest_filename)
        if statinfo[6] == expected_bytes:
            print('Found and verified', dest_filename)
            return
        
        if force or not os.path.exists(dest_filename):
            print('Attempting to download:', dest_filename) 
            filename, _ = urlretrieve(WEIGHTS_URL, dest_filename, reporthook=self._download_progress_hook)
            print('\nDownload Complete!')
        else:
            raise Exception('Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    
        return dest_filename


    def _download_progress_hook(self, count, blockSize, totalSize):
        percent = int(count * blockSize * 100 / totalSize)

        if self.last_percent_reported != percent:
            if percent % 5 == 0:
                print('{}%'.format(percent), end='')
            else:
                print('.', end='')
        
        self.last_percent_reported = percent


    def _create(self):
        Y = self._conv(self._X, filters=96, kernel_size=(11, 11), strides=(4, 4), group=1, padding='VALID', layer_index=1)
        Y = self._lrn(Y, layer_index=1)
        Y = self._max_pool(Y, layer_index=1)

        Y = self._conv(Y, filters=256, kernel_size=(5, 5), strides=(1, 1), group=2, layer_index=2)
        Y = self._lrn(Y, layer_index=2)
        Y = self._max_pool(Y, layer_index=2)

        Y = self._conv(Y, filters=384, kernel_size=(3, 3), strides=(1, 1), group=1, layer_index=3)

        Y = self._conv(Y, filters=384, kernel_size=(3, 3), strides=(1, 1), group=2, layer_index=4)

        Y = self._conv(Y, filters=256, kernel_size=(3, 3), strides=(1, 1), group=2, layer_index=5)
        Y = self._max_pool(Y, layer_index=5)
        Y = self._flatten(Y, layer_index=5)

        Y = self._fc(Y, output_num=4096, is_relu=True, layer_index=6)
        Y = self._dropout(Y, keep_prob=self._keep_prob, layer_index=6)

        Y = self._fc(Y, output_num=4096, is_relu=True, layer_index=7)
        Y = self._dropout(Y, keep_prob=self._keep_prob, layer_index=7)

        Y = self._fc(Y, output_num=self._classes_num, is_relu=False, layer_index=8)
        output_Z = Y
        Y = self._softmax(Y, layer_index=8)
        output_A = Y

        return (output_Z, output_A)


    def interface(self):
        return {
            'input_placeholder': self._X,
            'keep_prob_placeholder': self._keep_prob,
            'output_Z': self._output_Z,
            'output_A': self._output_A
        }


    def _init_placeholders(self, image_shape):
        with tf.name_scope('placeholders'):
            X = tf.placeholder(dtype=tf.float32, shape=(None, image_shape[0], image_shape[1], image_shape[2]))
            keep_prob = tf.placeholder(dtype=tf.float32)

        return (X, keep_prob)


    def _conv(self, input, filters, kernel_size, strides, layer_index, group=1, padding='SAME'):
        input_num = input.shape[-1].value
        trainable = False if layer_index in self._freeze_layer_indexes else True

        with tf.variable_scope('conv' + str(layer_index)):
            W = tf.get_variable(
                shape=(kernel_size[0], kernel_size[1], input_num/group, filters),
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                trainable=trainable,
                name='weights'
            )

            b = tf.get_variable(
                shape=(filters),
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=trainable,
                name='bias'
            )

            if group == 1:
                Z = tf.nn.conv2d(input, W, strides=(1, strides[0], strides[1], 1), padding=padding) + b
            else:
                input_group = tf.split(value=input, num_or_size_splits=group, axis=3)
                W_group = tf.split(value=W, num_or_size_splits=group, axis=3)
                b_group = tf.split(value=b, num_or_size_splits=group, axis=0)

                Z_group = []
                for i in range(group):
                    Z_part = tf.nn.conv2d(
                        input_group[i],
                        W_group[i],
                        strides=(1, strides[0], strides[1], 1),
                        padding=padding
                    ) + b_group[i]

                    Z_group.append(Z_part)

                Z = tf.concat(Z_group, axis=3)

            A = tf.nn.relu(Z)

        return A


    def _fc(self, input, output_num, layer_index, is_relu=True):
        input_num = input.shape[-1].value
        trainable = False if layer_index in self._freeze_layer_indexes else True

        with tf.variable_scope('fc' + str(layer_index)):
            W = tf.get_variable(
                shape=(input_num, output_num),
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=trainable,
                name='weights'
            )

            b = tf.get_variable(
                shape=(output_num),
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=trainable,
                name='bias'
            )

            Z = tf.matmul(input, W) + b

            if is_relu:
                A = tf.nn.relu(Z)

        if is_relu:
            return A
        else:
            return Z


    def _max_pool(self, input, layer_index):
        with tf.name_scope('max_pool' + str(layer_index)):
            output = tf.nn.max_pool(input, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')

        return output


    def _dropout(self, input, layer_index, keep_prob=1):
        with tf.name_scope('dropout' + str(layer_index)):
            output = tf.nn.dropout(input, keep_prob=keep_prob)

        return output


    def _lrn(self, input, layer_index):
        with tf.name_scope('lrn' + str(layer_index)):
            output = tf.nn.lrn(input, depth_radius=2, alpha=1e-05, beta=0.75, bias=1.0)

        return output


    def _flatten(self, input, layer_index):
        with tf.name_scope('flatten' + str(layer_index)):
            output = tf.reshape(input, (-1, input.shape[1].value * input.shape[2].value * input.shape[3].value))

        return output


    def _softmax(self, input, layer_index):
        with tf.name_scope('softmax' + str(layer_index)):
            output = tf.nn.softmax(input, name='softmax')

        return  output


    def _relu(self, input, layer_index):
        with tf.name_scope('relu' + str(layer_index)):
            output = tf.nn.relu(input)

        return output

