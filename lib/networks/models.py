# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

import numpy as np
import tensorflow as tf
from networks.network import Network
from networks import losses
from fast_rcnn.config import cfg
import networks.net_utils as utils
import tensorflow.contrib.slim as slim
from networks.basenets import *

"""
A TensorFlow implementation of the scene graph generation models introduced in
"Scene Graph Generation by Iterative Message Passing" by Xu et al.
"""

class vrdnet(Network):
    def __init__(self, data):
        # self.inputs = self._init_inputs()
        self.data = data
        self.ims = data['ims']
        self.rois = data['rois']
        self.labels = self.data['labels']
        self.rel_rois = data['rel_rois']
        self.num_roi = data['num_roi']
        self.num_rel = data['num_rel']
        self.num_classes = data['num_classes']
        self.num_predicates = data['num_predicates']
        self.relations = data['relations']
        self.predicates = data['predicates']
        self.is_training = data['is_training']
        self.iterable = False
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = {}
        self.if_pred_rel = data['if_pred_rel'] if 'if_pred_rel' in data else True
        self.if_pred_cls = data['if_pred_cls'] if 'if_pred_cls' in data else False
        self.if_pred_bbox = data['if_pred_bbox'] if 'if_pred_bbox' in data else False
        self.use_gt_box = data['use_gt_box'] if 'use_gt_box' in data else True
        self.use_gt_cls = data['use_gt_cls'] if 'use_gt_cls' in data else True
        self.stop_gradient = data['stop_gradient'] if 'stop_gradient' in data else False
        self.use_vis = data['use_vis'] if 'use_vis' in data else False
        self.use_spatial = data['use_spatial'] if 'use_spatial' in data else False
        self.use_class = data['use_class'] if 'use_class' in data else False
        self.use_embedding = data['use_embedding'] if 'use_embedding' in data else False
        self.embedded_size = data['embedded_size'] if 'embedded_size' in data else 64
        self.if_weight_reg = cfg.TRAIN.WEIGHT_REG
        self.roi_scale = 1.0/16
        self.loss_weights = { 'rel': 1, 'cls': 1, 'bbox': 1 }
        self.pooling_size = 7

    @classmethod
    def inputs(cls, num_classes, prior_size, obj_embedding_size, is_training=False):
        input_pls = {
            'ims': tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]),
            'rois': tf.placeholder(dtype=tf.float32, shape=[None, 5]),
            'rel_rois': tf.placeholder(dtype=tf.float32, shape=[None, 5]),
            'labels': tf.placeholder(dtype=tf.int32, shape=[None]),
            # 'bboxes': tf.placeholder(dtype=tf.float32, shape=[None, 4]),
            'relations': tf.placeholder(dtype=tf.int32, shape=[None, 2]),
            'predicates': tf.placeholder(dtype=tf.int32, shape=[None]),
            'rel_spts': tf.placeholder(dtype=tf.int32, shape=[None]),
            'bbox_targets': tf.placeholder(dtype=tf.float32, shape=[None, 4 * num_classes]),
            'bbox_inside_weights': tf.placeholder(dtype=tf.float32, shape=[None, 4 * num_classes]),
            'num_roi': tf.placeholder(dtype=tf.int32, shape=[]),  # number of rois per batch
            'num_rel': tf.placeholder(dtype=tf.int32, shape=[]),  # number of relationships per batch
            'obj_context_o': tf.placeholder(dtype=tf.int32, shape=[None]),
            'obj_context_p': tf.placeholder(dtype=tf.int32, shape=[None]),
            'obj_context_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
            'rel_context': tf.placeholder(dtype=tf.int32, shape=[None]),
            'rel_context_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
            'obj_embedding': tf.placeholder(dtype=tf.float32, shape=[num_classes, obj_embedding_size]),
            'prior': tf.placeholder(dtype=tf.float32, shape=[None, prior_size]),
            'obj_matrix': tf.placeholder(dtype=tf.float32, shape=[None, None]),
            'rel_matrix': tf.placeholder(dtype=tf.float32, shape=[None, None]),
            'rel_weight_labels': tf.placeholder(dtype=tf.int32, shape=[None]),
            'rel_weight_rois': tf.placeholder(dtype=tf.float32, shape=[None, 5]),
            'rel_triple_inds': tf.placeholder(dtype=tf.int32, shape=[None, 2]),
            'rel_triple_labels': tf.placeholder(dtype=tf.int32, shape=[None, 1]),
        }
        return input_pls

    def setup(self):
        self._net()

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], 7, 7, name='roi_conv_out')
        roi_fc_out = self._net_roi_fc(roi_conv_out)

        if self.if_pred_cls:
            self._cls_pred(roi_fc_out)
        if self.if_pred_bbox:
            self._bbox_pred(roi_fc_out)
        if self.if_pred_rel:
            rel_roi_conv_roi = self._net_roi_pooling([conv_net, self.rel_rois], 7, 7, name='rel_roi_conv_out')
            rel_roi_flatten = self._net_conv_reshape(rel_roi_conv_roi, name='rel_roi_flatten')
            rel_roi_fc_out = self._net_rel_roi_fc(rel_roi_flatten)
            rel_roi_final = rel_roi_fc_out
            rel_inx1, rel_inx2 = self._relation_indexes()
            if self.use_context:
                con_sub = tf.gather(roi_fc_out, rel_inx1)
                con_obj = tf.gather(roi_fc_out, rel_inx2)
                context = slim.fully_connected(tf.concat([con_sub, con_obj], axis=1), 4096, scope="context")
                rel_roi_final = slim.fully_connected(tf.concat([rel_roi_fc_out, context], axis=1), 4096)
            if self.use_spatial:
                spatial_feat = self._spatial_feature(rel_inx1, rel_inx2)
                rel_roi_final = tf.concat([rel_roi_final, spatial_feat], axis=1)
            if self.use_class:
                class_feat = self._class_feature(rel_inx1, rel_inx2)
                rel_roi_final = tf.concat([rel_roi_final, class_feat], axis=1)
            # rel_roi_final = slim.fully_connected(rel_roi_final, 512)
            self._rel_pred(rel_roi_final)

    def _rel_visual_feature(self, conv_layer, ):
        pass

    def _class_feature(self, rel_inx1, rel_inx2):
        if not self.if_pred_cls:
            # use ground-true boxes
            class_prob = tf.one_hot(self.labels, depth=self.num_classes)
        else:
            raise NotImplementedError()
        if self.use_embedding:
            obj_emb = tf.Variable(tf.zeros(shape=self.data['obj_embedding'].shape.as_list(), dtype=tf.float32), trainable=False, name='obj_emb')
            self.embedding_init = obj_emb.assign(self.data['obj_embedding'])
            self.cls_emb = tf.nn.embedding_lookup(obj_emb, self.labels, name='cls_emb')
        else:
            self.cls_emb = slim.fully_connected(class_prob, self.embedded_size, scope='cls_emb')
            # raise NotImplementedError()
        cls_emb_sub = tf.gather(self.cls_emb, rel_inx1)
        cls_emb_obj = tf.gather(self.cls_emb, rel_inx2)
        return cls_emb_sub, cls_emb_obj

    def local_attention(self, context, conv_feat, name='local_attention', reuse = False):
        with tf.variable_scope(name, 'local_attention', reuse=reuse):
            conv_shape = conv_feat.get_shape()
            conv_feat = tf.reshape(conv_feat, [-1, conv_shape[1]*conv_shape[2], conv_shape[3]])
            cont_resp = tf.reshape(tf.tile(context, [1, conv_shape[1]*conv_shape[2]]), shape=[-1, conv_shape[1]*conv_shape[2], context.get_shape().as_list()[1]])
            input = tf.concat([conv_feat, cont_resp], axis=2)
            net = slim.fully_connected(input, 2048, scope='att1')
            net = slim.dropout(net, keep_prob=self.keep_prob)
            net = slim.fully_connected(net, 2048, scope='att2')
            net = slim.dropout(net, keep_prob=self.keep_prob)
            net = slim.fully_connected(net, 1, scope='att3')
            weights = tf.squeeze(net, axis=2)
            weights = tf.nn.softmax(weights)
            weights = tf.reshape(tf.tile(weights, [1, conv_feat.get_shape()[2]]), [-1, conv_feat.get_shape()[1], conv_feat.get_shape()[2]])
            net = tf.reduce_mean(weights * conv_feat, axis=1)
            # net = tf.reshape(net, [-1, conv_shape[1], conv_shape[2], conv_shape[3]])
        return net

    def _normalize_bbox(self, bbox):
        wi = tf.cast(tf.shape(self.ims)[2], tf.float32)
        hi = tf.cast(tf.shape(self.ims)[1], tf.float32)

        areai = wi * hi
        w = bbox[:, 2:3] - bbox[:, 0:1]
        h = bbox[:, 3:4] - bbox[:, 1:2]
        area = w * h
        nx = bbox[:, 0:1] / wi
        ny = bbox[:, 1:2] / hi
        nw = bbox[:, 2:3] / wi
        nh = bbox[:, 3:4] / hi
        na = area / areai
        return tf.concat([nx, ny, nw, nh, na], axis=1, name='norm_bbox')

    def _spatial_feature(self, rel_inx1, rel_inx2):
        if not self.if_pred_cls:
            # use ground-true boxes
            bbox = self.rois[:, 1:5]
        else:
            raise NotImplementedError()
        norm_bbox = self._normalize_bbox(bbox)
        bbox_sub = tf.gather(bbox, rel_inx1)
        bbox_obj = tf.gather(bbox, rel_inx2)
        norm_bbox_sub = tf.gather(norm_bbox, rel_inx1)
        norm_bbox_obj = tf.gather(norm_bbox, rel_inx2)
        rel_spt = self._relative_spatial(bbox_sub, bbox_obj)
        spatial = tf.concat([norm_bbox_sub, norm_bbox_obj, rel_spt], axis=1, name='spatial_feature')
        # boxes
        return spatial

    def _relative_spatial(self, sub, obj):
        w = sub[:, 2] - sub[:, 0]
        h = sub[:, 3] - sub[:, 1]
        w_ = obj[:, 2] - obj[:, 0]
        h_ = obj[:, 3] - obj[:, 1]
        tx = tf.expand_dims((sub[:, 0] - obj[:, 0]) / w_, 1)
        ty = tf.expand_dims((sub[:, 1] - obj[:, 1]) / h_, 1)
        tw = tf.expand_dims(tf.log(w/w_), 1)
        th = tf.expand_dims(tf.log(h/h_), 1)
        return tf.concat([tx, ty, tw, th], axis=1)

    def _spatial_mask1(self, rel_inx, rel_conv, suffix = '', reuse = False):
        def get_maps(x1, y1, x2, y2, shape):
            maps = np.zeros(shape=shape, dtype=np.float32)
            for i in range(shape[0]):
                # print(i, y1[i], y2[i], x1[i], x2[i])
                w, h = x2[i] - x1[i], y2[i] - y1[i]
                # if w <= 0: w = 1
                # if h <= 0: h = 1
                maps[i, y1[i]:y1[i]+h, x1[i]:x1[i]+w] = 1.0
            return maps
        rel_box = self.rel_rois
        if not self.if_pred_cls:
            # use ground-true boxes
            bbox = self.rois
        else:
            raise NotImplementedError()
        bbox_sub = tf.gather(bbox, rel_inx)
        W, H = rel_box[:, 3] - rel_box[:, 1], rel_box[:, 4] - rel_box[:, 2]
        x1 = tf.cast(tf.floor((bbox_sub[:, 1] - rel_box[:, 1]) * self.pooling_size / W), tf.int32)
        y1 = tf.cast(tf.floor((bbox_sub[:, 2] - rel_box[:, 2]) * self.pooling_size / H), tf.int32)
        x2 = tf.cast(tf.ceil((bbox_sub[:, 3] - rel_box[:, 1]) * self.pooling_size / W), tf.int32)
        y2 = tf.cast(tf.ceil((bbox_sub[:, 4] - rel_box[:, 2]) * self.pooling_size / H), tf.int32)
        maps = tf.py_func(get_maps, [x1, y1, x2, y2, tf.shape(rel_conv)[0:3]], tf.float32)
        return tf.expand_dims(maps, -1)

    def _spatial_conv(self, rel_inx, roi_conv, rel_conv):
        def get_maps(x1, y1, x2, y2, roi_conv, shape):
            import cv2
            maps = np.zeros(shape=shape, dtype=np.float32)
            for i in range(shape[0]):
                # print(i, roi_conv[i].shape, x2[i] - x1[i], y2[i] - y1[i])
                if x2[i]>self.pooling_size: x2[i] = self.pooling_size
                if y2[i]>self.pooling_size: y2[i] = self.pooling_size
                w, h = x2[i] - x1[i], y2[i] - y1[i]
                # if w <= 0: w = 1
                # if h <= 0: h = 1
                val = cv2.resize(roi_conv[i], (w, h))
                # print("**", x1[i],x2[i],y1[i],y2[i], w, h, val.shape, maps[i, y1[i]:y2[i], x1[i]:x2[i], :].shape)
                maps[i, y1[i]:y2[i], x1[i]:x2[i], :] = val
            return maps
        rel_box = self.rel_rois
        if not self.if_pred_cls:
            # use ground-true boxes
            bbox = self.rois
        else:
            bbox = self.rois
            # raise NotImplementedError()
        bbox_sub = tf.gather(bbox, rel_inx)
        # W, H = rel_box[:, 3:4] - rel_box[:, 1:2], rel_box[:, 4:5] - rel_box[:, 2:3]
        # x1 = tf.cast(tf.floor(bbox_sub[:, 1:2] * self.pooling_size / W - rel_box[:, 1:2]), tf.int32)
        # y1 = tf.cast(tf.floor(bbox_sub[:, 2:3] * self.pooling_size / H - rel_box[:, 2:3]), tf.int32)
        # x2 = tf.cast(tf.ceil(bbox_sub[:, 3:4] * self.pooling_size / W - rel_box[:, 1:2]), tf.int32)
        # y2 = tf.cast(tf.ceil(bbox_sub[:, 4:5] * self.pooling_size / H - rel_box[:, 2:3]), tf.int32)
        W, H = rel_box[:, 3] - rel_box[:, 1], rel_box[:, 4] - rel_box[:, 2]
        x1 = tf.cast(tf.floor((bbox_sub[:, 1] - rel_box[:, 1]) * self.pooling_size / W), tf.int32)
        y1 = tf.cast(tf.floor((bbox_sub[:, 2] - rel_box[:, 2]) * self.pooling_size / H), tf.int32)
        x2 = tf.cast(tf.ceil((bbox_sub[:, 3] - rel_box[:, 1]) * self.pooling_size / W), tf.int32)
        y2 = tf.cast(tf.ceil((bbox_sub[:, 4] - rel_box[:, 2]) * self.pooling_size / H), tf.int32)
        # w = tf.cast(tf.ceil((bbox_sub[:, 3] - bbox_sub[:, 1]) * self.pooling_size / W), tf.int32)
        # h = tf.cast(tf.ceil((bbox_sub[:, 4] - bbox_sub[:, 2]) * self.pooling_size / H), tf.int32)
        maps = tf.py_func(get_maps, [x1, y1, x2, y2, roi_conv, tf.shape(rel_conv)], tf.float32)
        return maps

    def _relation_indexes(self):
        assert self.relations.get_shape().as_list()[1] == 2
        self.rel_inx1, self.rel_inx2 = tf.split(self.relations, num_or_size_splits=2, axis=1)
        return tf.squeeze(self.rel_inx1, axis=1), tf.squeeze(self.rel_inx2, axis=1)

    def _net_conv(self, inputs):
        net = slim.conv2d(inputs, 64, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool4')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')
        if self.stop_gradient: net = tf.stop_gradient(net, name='stop_gradient')
        return net

    def _net_roi_pooling(self, inputs, pooled_height, pooled_width, name=None):
        assert len(inputs) == 2
        net = self.roi_pool(inputs, pooled_height, pooled_width, self.roi_scale, name)
        net[0].set_shape([None, pooled_height, pooled_width, inputs[0].get_shape().as_list()[3]])
        return net[0]

    def _net_crop_pooling(self, inputs, pooling_size, name):
        return self.crop_pool(inputs[0], inputs[1], pooling_size, 1.0/self.roi_scale, name)

    def _net_conv_reshape(self, inputs, name=None):
        shape = inputs.get_shape().as_list()
        dim = shape[1]*shape[2]*shape[3]
        net = tf.reshape(inputs, shape=[-1, dim], name=name)
        return net

    def _net_roi_fc(self, inputs):
        net = slim.fully_connected(inputs, 4096, scope='fc6')
        net = slim.dropout(net, self.keep_prob, scope='drop6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, self.keep_prob, scope='roi_fc_out')
        return net

    def _net_rel_roi_fc(self, inputs):
        net = slim.fully_connected(inputs, 4096, scope='rel_fc6')
        net = slim.dropout(net, self.keep_prob, scope='rel_drop6')
        net = slim.fully_connected(net, 4096, scope='rel_fc7')
        net = slim.dropout(net, self.keep_prob, scope='rel_roi_fc_out')
        return net

    # predictions
    def _cls_pred(self, inputs, layer_suffix='', reuse=False, new_var=False):
        layer_name = 'cls_score'+layer_suffix if new_var else 'cls_score'
        net = slim.fully_connected(inputs, self.num_classes, activation_fn=None, reuse=reuse, scope=layer_name)
        self.layers[layer_name] = net
        net = slim.softmax(net, scope='cls_prob'+layer_suffix)
        self.layers['cls_prob'+layer_suffix] = net
        net = tf.argmax(net, axis=1, name='cls_pred'+layer_suffix)
        self.layers['cls_pred' + layer_suffix] = net
        return net

    def _bbox_pred(self, inputs, layer_suffix='', reuse=False, new_var=False):
        layer_name = 'bbox_pred'+layer_suffix if new_var else 'bbox_pred'
        net = slim.fully_connected(inputs, self.num_classes*4, activation_fn=None, reuse=reuse, scope=layer_name)
        self.layers[layer_name] = net
        return net

    def _rel_pred(self, inputs, layer_suffix='', reuse=False, new_var=False):
        layer_name = 'rel_score'+layer_suffix
        net = slim.fully_connected(inputs, self.num_predicates, activation_fn=None, reuse=reuse, scope=layer_name)
        self.layers[layer_name] = net
        net = slim.softmax(net, scope='rel_prob' + layer_suffix)
        self.layers['rel_prob' + layer_suffix] = net
        net = tf.argmax(net, axis=1, name='rel_pred' + layer_suffix)
        self.layers['rel_pred' + layer_suffix] = net
        return net

    # Losses
    def losses(self):
        losses = {}
        losses['loss_total'] = None
        if self.if_weight_reg:
            l_reg = tf.losses.get_regularization_losses()
            if len(l_reg):
                losses['loss_total'] = losses['loss_reg'] = tf.add_n(l_reg)
        if self.if_pred_rel:
            self._rel_losses(losses)
            if losses['loss_total'] == None:
                losses['loss_total'] = losses['loss_rel']
            else:
                losses['loss_total'] += losses['loss_rel']
        if self.if_pred_cls:
            self._cls_losses(losses)
            if losses['loss_total'] == None:
                losses['loss_total'] = losses['loss_cls']
            else:
                losses['loss_total'] = tf.add(losses['loss_total'], losses['loss_cls'])
        if self.if_pred_bbox:
            self._bbox_losses(losses)
            if losses['loss_total'] == None:
                losses['loss_total'] = losses['loss_bbox']
            else:
                losses['loss_total'] = tf.add(losses['loss_total'], losses['loss_bbox'])
        return losses

    def metrics(self, ops={}):
        if self.if_pred_cls:
            ops['acc_cls'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.layers['cls_prob'], axis=1, output_type=tf.int32), self.data['labels']), tf.float32))
        if self.if_pred_rel:
            ops['acc_rel'] = self.accuracy(self.layers['rel_pred'], self.predicates, "acc_rel", ignore_bg=cfg.TRAIN.USE_SAMPLE_GRAPH)

    def _rel_losses(self, ops={}, suffix=''):
        rel_score = self.get_output('rel_score'+suffix)
        # lo = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.data['predicates'], logits=rel_score)
        # ops['loss_rel' + suffix] =tf .reduce_mean(lo, name='rel_loss' + suffix)
        ops['loss_rel'+suffix] = self.sparse_softmax(rel_score, self.predicates,
                                                     name='rel_loss'+suffix, loss_weight=self.loss_weights['rel'], ignore_bg=cfg.TRAIN.USE_SAMPLE_GRAPH)
        return ops

    def _cls_losses(self, ops={}, suffix=''):
        # classification loss
        cls_score = self.get_output('cls_score'+suffix)
        ops['loss_cls'+suffix] = self.sparse_softmax(cls_score, self.data['labels'], name='cls_loss'+suffix, loss_weight=self.loss_weights['cls'])
        return ops

    def _bbox_losses(self, ops={}, suffix=''):
        # bounding box regression L1 loss
        if not self.use_gt_bbox:
            bbox_pred = self.get_output('bbox_pred'+suffix)
            ops['loss_bbox'+suffix]  = self.l1_loss(bbox_pred, self.data['bbox_targets'], 'reg_loss'+suffix,
                                              self.loss_weights['bbox'], self.data['bbox_inside_weights'])
        else:
            print('NO BBOX REGRESSION!!!!!')
        return ops

    def sparse_softmax(self, logits, labels, name, loss_weight=1, ignore_bg=False):
        labels = tf.cast(labels, dtype=tf.int32)
        batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        if ignore_bg:  # do not penalize background class
            loss_mask = tf.cast(tf.greater(labels, 0), tf.float32)
            batch_loss = tf.multiply(batch_loss, loss_mask)
        if ignore_bg:
            loss = tf.reduce_sum(batch_loss)/(tf.reduce_sum(loss_mask)+1)
        else:
            loss = tf.reduce_mean(batch_loss)
        loss = tf.multiply(loss, loss_weight, name=name)
        return loss

    def l1_loss(self, preds, targets, name, target_weights=None, loss_weight=1):
        l1 = tf.abs(tf.subtract(preds, targets))
        if target_weights is not None:
            l1 = tf.multiply(target_weights, l1)
        batch_loss = tf.reduce_sum(l1, axis=1)
        loss = tf.reduce_mean(batch_loss)
        loss = tf.multiply(loss, loss_weight, name=name)
        return loss

    def accuracy(self, pred, labels, name, ignore_bg=False):
        correct_pred = tf.cast(tf.equal(labels, tf.cast(pred, tf.int32)), tf.float32)
        if ignore_bg:  # ignore background
            mask = tf.cast(tf.greater(labels, 0), tf.float32)
            one = tf.constant([1], tf.float32)
            # in case zero foreground preds
            num_preds = tf.maximum(tf.reduce_sum(mask), one)
            acc_op = tf.squeeze(tf.div(tf.reduce_sum(tf.multiply(correct_pred, mask)), num_preds))
        else:
            acc_op = tf.reduce_mean(correct_pred)

        # override the name
        return tf.identity(acc_op, name=name)

    def cls_pred_output(self, iters=None):
        if iters is not None:
            op = {}
            for i in iters:
                if self.iterable and i != self.n_iter - 1:
                    op[i] = self.get_output('cls_prob_iter%i' % i)
                else:
                    op[i] = self.get_output('cls_prob')

        else:
            op = self.get_output('cls_prob')
        return op

    def bbox_pred_output(self, iters=None):
        if iters is not None:
            op = {}
            for i in iters:
                op[i] = self.get_output('bbox_pred')

        else:
            op = self.get_output('bbox_pred')
        return op

    def rel_pred_output(self):
        return self.get_output('rel_prob')

class basenet(vrdnet):
    def __init__(self, data):
        super(basenet, self).__init__(data)
        self.model = self.build_base(data['basenet']) if 'basenet' in data else self.build_base('vgg16')
        self.roi_scale = self.model.roi_scale
        self._variables_to_fix = {}
        self._scope = self.model._scope

    def build_base(self, network_name):
        if(network_name=='vgg16'):
            return vgg16net(self.keep_prob, self.stop_gradient)
        elif(network_name=='res50'):
            return resnetv1(is_training=not self.stop_gradient, num_layers=50)
        else:
            raise NotImplementedError

    def _net_conv(self, inputs):
        return self.model._net_conv(inputs)

    def _net_roi_fc(self, inputs, reuse=False):
        return self.model._net_roi_fc(inputs, reuse=reuse)

    def get_variables_to_restore(self):
        variables = tf.global_variables(scope=self._scope)
        return variables

    def get_variables_to_restore_imagenet(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == (self._scope + '/fc6/weights:0') or \
                            v.name == (self._scope + '/fc7/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables_imagenet(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16') as scope:
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv,
                                              self._scope + "/fc7/weights": fc7_conv,
                                              self._scope + "/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                                      self._variables_to_fix[
                                                                                                          self._scope + '/fc6/weights:0'].get_shape())))
                # print(self.image)
                # print(self._variables_to_fix[self._scope + '/fc6/weights:0'].get_shape())
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                                      self._variables_to_fix[
                                                                                                          self._scope + '/fc7/weights:0'].get_shape())))
                # print(self._variables_to_fix[self._scope + '/fc7/weights:0'].get_shape())
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))

