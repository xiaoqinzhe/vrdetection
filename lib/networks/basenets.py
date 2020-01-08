import tensorflow as tf
from fast_rcnn.config import cfg
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block

class vgg16net:
    def __init__(self, keep_prob=0.5, stop_gradient=False):
        self.keep_prob=keep_prob
        self.stop_gradient = stop_gradient
        self.roi_scale = 1.0/16
        self._variables_to_fix = {}
        self._scope = 'vgg_16'

    def _net_conv(self, inputs):
        with tf.variable_scope(self._scope):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            if self.stop_gradient: net = tf.stop_gradient(net, name='stop_gradient')
        return net

    def _net_roi_fc(self, inputs, reuse=False):
        with tf.variable_scope(self._scope, reuse=reuse):
            flatten = slim.flatten(inputs, scope='flatten')
            net = slim.fully_connected(flatten, 4096, scope='fc6')
            net = slim.dropout(net, self.keep_prob, scope='drop6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, self.keep_prob, scope='roi_fc_out')
            if self.stop_gradient: net = tf.stop_gradient(net, name='stop_gradient')
        return net

def resnet_arg_scope(is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
        weights_initializer=slim.variance_scaling_initializer(),
        trainable=is_training,
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

class resnetv1:
    def __init__(self, is_training, num_layers=50):
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self.roi_scale=self._feat_compress[0]
        self._num_layers = num_layers
        self._scope = 'resnet_v1_%d' % num_layers
        self._decide_blocks()
        self.is_training=is_training

    def _net_conv(self, inputs):
        return self._image_to_head(inputs, self.is_training)

    def _net_roi_fc(self, inputs, reuse=False):
        return self._head_to_tail(inputs, self.is_training, reuse=reuse)

    # Do the first few layers manually, because 'SAME' padding can behave inconsistently
    # for images of different sizes: sometimes 0, sometimes 1
    def _build_base(self, image):
        with tf.variable_scope(self._scope, self._scope):
            net = resnet_utils.conv2d_same(image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
        return net

    def _image_to_head(self, image, is_training, reuse=None):
        assert (0 <= cfg.RESNET.FIXED_BLOCKS <= 3)
        # Now the base is always fixed during training
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
          net_conv = self._build_base(image)
        if cfg.RESNET.FIXED_BLOCKS > 0:
          with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                               self._blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                               global_pool=False,
                                               include_root_block=False,
                                               reuse=reuse,
                                               scope=self._scope)
        if cfg.RESNET.FIXED_BLOCKS < 3:
          with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                               self._blocks[cfg.RESNET.FIXED_BLOCKS:-1],
                                               global_pool=False,
                                               include_root_block=False,
                                               reuse=reuse,
                                               scope=self._scope)
        return net_conv

    def _head_to_tail(self, pool5, is_training, reuse=None):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
          fc7, _ = resnet_v1.resnet_v1(pool5,
                                       self._blocks[-1:],
                                       global_pool=False,
                                       include_root_block=False,
                                       reuse=reuse,
                                       scope=self._scope)
          # average pooling done by reduce_mean
          fc7 = tf.reduce_mean(fc7, axis=[1, 2])
        return fc7

    def _decide_blocks(self):
        # choose different blocks for different number of layers
        if self._num_layers == 50:
          self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                          resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                          # use stride 1 for the last conv4 layer
                          resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
                          resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        elif self._num_layers == 101:
          self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                          resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                          # use stride 1 for the last conv4 layer
                          resnet_v1_block('block3', base_depth=256, num_units=23, stride=1),
                          resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        elif self._num_layers == 152:
          self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                          resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
                          # use stride 1 for the last conv4 layer
                          resnet_v1_block('block3', base_depth=256, num_units=36, stride=1),
                          resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        else:
          # other numbers are not supported
          raise NotImplementedError