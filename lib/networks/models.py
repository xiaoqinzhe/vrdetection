# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

import numpy as np
import tensorflow as tf
from network import Network
import losses
from fast_rcnn.config import cfg
import net_utils as utils
import tensorflow.contrib.slim as slim


"""
A TensorFlow implementation of the scene graph generation models introduced in
"Scene Graph Generation by Iterative Message Passing" by Xu et al.
"""

class vrdnet(Network):
    def __init__(self, data):
        self.inputs = []
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
        self.iterable = False
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = {}
        self.if_pred_rel = data['if_pred_rel'] if 'if_pred_rel' in data else True
        self.if_pred_cls = data['if_pred_cls'] if 'if_pred_cls' in data else False
        self.if_pred_bbox = data['if_pred_bbox'] if 'if_pred_bbox' in data else False
        self.use_gt_box = data['use_gt_box'] if 'use_gt_box' in data else True
        self.use_gt_cls = data['use_gt_cls'] if 'use_gt_cls' in data else True
        self.stop_gradient = data['stop_gradient'] if 'stop_gradient' in data else False
        self.use_context = data['use_context'] if 'use_context' in data else False
        self.use_spatial = data['use_spatial'] if 'use_spatial' in data else False
        self.use_class = data['use_class'] if 'use_class' in data else False
        self.use_embedding = data['use_embedding'] if 'use_embedding' in data else False
        self.embedded_size = data['embedded_size'] if 'embedded_size' in data else 300
        self.if_weight_reg = cfg.TRAIN.WEIGHT_REG
        self.roi_scale = 1.0/16
        self.loss_weights = { 'rel': 1, 'cls': 1, 'bbox': 1 }
        self.pooling_size = 7

    def setup(self):
        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                       biases_initializer=tf.constant_initializer(0.0)):
            self._net()

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], 7, 7, name='roi_conv_out')
        roi_flatten = self._net_conv_reshape(roi_conv_out, name='roi_flatten')
        roi_fc_out = self._net_roi_fc(roi_flatten)

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
            cls_emb = tf.nn.embedding_lookup(self.data['obj_embedding'], self.labels, name='cls_emb')
        else:
            cls_emb = slim.fully_connected(class_prob, self.embedded_size, scope='cls_emb')
            # raise NotImplementedError()
        cls_emb_sub = tf.gather(cls_emb, rel_inx1)
        cls_emb_obj = tf.gather(cls_emb, rel_inx2)
        return cls_emb_sub, cls_emb_obj

    def _spatial_feature(self, rel_inx1, rel_inx2):
        if not self.if_pred_cls:
            # use ground-true boxes
            bbox = self.rois[:, 1:5]
        else:
            raise NotImplementedError()
        bbox_sub = tf.gather(bbox, rel_inx1)
        bbox_obj = tf.gather(bbox, rel_inx2)
        spatial = self._relative_spatial(bbox_sub, bbox_obj)
        spatial = tf.identity(spatial, name="spatial_feat")
        # boxes

        return spatial

    def _relative_spatial(self, sub, obj):
        w = sub[:, 2] - sub[:, 0]
        h = sub[:, 3] - sub[:, 1]
        w_ = obj[:, 2] - obj[:, 0]
        h_ = obj[:, 3] - obj[:, 1]
        tx = tf.expand_dims((sub[:, 0] - obj[:, 0]) / w_, 1)
        ty = tf.expand_dims((sub[:, 1] - obj[:, 1]) / w_, 1)
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
        rel1, rel2 = tf.split(self.relations, num_or_size_splits=2, axis=1)
        return tf.squeeze(rel1, axis=1), tf.squeeze(rel2, axis=1)

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
            losses['loss_total'] = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
        if self.if_pred_rel:
            self._rel_losses(losses)
            losses['loss_total'] = losses['loss_rel']
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
            loss = tf.reduce_sum(batch_loss)/tf.reduce_sum(loss_mask)
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

    def rel_pred_output(self, iters=None):
        if iters is not None:
            op = {}
            if type(iters) == str:
                op[0] = self.get_output('rel_prob'+iters)
                return op
            for i in iters:
                if self.iterable and i != self.n_iter - 1:
                    op[i] = self.get_output('rel_prob_iter%i' % i)
                else:
                    op[i] = self.get_output('rel_prob')

        else:
            op = self.get_output('rel_prob')
        return op

class sptnet(vrdnet):
    def __init__(self, data):
        super(sptnet, self).__init__(data)
        self.num_spatials = data['num_spatials']
        self.if_pred_spt = data['if_pred_spt'] if 'if_pred_spt' in data else True
        self.if_pred_rel = True
        self.loss_weights['spt'] = 1.0

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], 7, 7, name='roi_conv_out')
        roi_flatten = self._net_conv_reshape(roi_conv_out, name='roi_flatten')
        roi_fc_out = self._net_roi_fc(roi_flatten)
        rel_roi_conv_roi = self._net_roi_pooling([conv_net, self.rel_rois], 7, 7, name='rel_roi_conv_out')
        rel_roi_flatten = self._net_conv_reshape(rel_roi_conv_roi, name='rel_roi_flatten')
        rel_roi_fc_out = self._net_rel_roi_fc(rel_roi_flatten)
        # spatial info
        bbox = self.rois[:, 1:5]
        if self.if_pred_spt:
            rel_inx1, rel_inx2 = self._relation_indexes()
            con_sub = tf.gather(roi_fc_out, rel_inx1)
            con_obj = tf.gather(roi_fc_out, rel_inx2)
            sub_feat = tf.concat([con_sub, rel_roi_fc_out], axis=1)
            obj_feat = tf.concat([con_obj, rel_roi_fc_out], axis=1)
            # sub_feat, obj_feat = con_sub, con_obj
            # feat = tf.concat([con_sub, con_obj], axis=1)
            # self._spatial_pred(feat)
            sub_feat = slim.fully_connected(sub_feat, 512)
            proj_sub = slim.fully_connected(sub_feat, 3)
            obj_feat = slim.fully_connected(obj_feat, 512)
            proj_obj = slim.fully_connected(obj_feat, 3)
            self._spatial_pred(proj_sub - proj_obj)
        if self.if_pred_rel:
            # class me
            cls_pred = tf.one_hot(self.labels, depth=self.num_classes)
            cls_fc = slim.fully_connected(cls_pred, 256, scope="cls_emb")
            cls_sub = tf.gather(cls_fc, rel_inx1)
            cls_obj = tf.gather(cls_fc, rel_inx2)
            rel_feat = tf.concat([con_sub, con_obj, rel_roi_fc_out, cls_sub, cls_obj, proj_sub - proj_obj], axis=1)
            rel_feat = slim.fully_connected(rel_feat, 512)
            rel_feat = slim.fully_connected(rel_feat, 512)
            self._rel_pred(rel_feat)

    def _spatial_pred(self, inputs):
        net = slim.fully_connected(inputs, self.num_spatials, scope='spt_score')
        self.layers['spt_score'] = net
        net = slim.softmax(net, scope='spt_prob')
        self.layers['spt_prob'] = net
        net = tf.argmax(net, axis=1, name='spt_pred')
        self.layers['spt_pred'] = net
        return net

    def _spatial_loss(self, ops={}):
        spt_score = self.get_output('spt_score')
        print(spt_score.get_shape(), self.data['rel_spts'].get_shape())
        ops['loss_spt'] = self.sparse_softmax(spt_score, self.data['rel_spts'],
                                                name='spt_loss', ignore_bg=True)
        # ops['loss_rel'+suffix] = losses.sparse_softmax(rel_score, self.data['predicates'],
        #                                         name='rel_loss'+suffix, ignore_bg=False)
        return ops

    def losses(self):
        losses = super(sptnet, self).losses()
        if self.if_pred_spt:
            self._spatial_loss(losses)
            if losses['loss_total'] == None:
                losses['loss_total'] = self.loss_weights['spt'] * losses['loss_spt']
            else:
                losses['loss_total'] = tf.add(losses['loss_total'], self.loss_weights['spt'] * losses['loss_spt'])
        return losses

    def metrics(self, ops={}):
        super(sptnet, self).metrics(ops)
        if self.if_pred_spt:
            # ops['acc_spt'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.layers['spt_prob'], axis=1, output_type=tf.int32), self.data['rel_spts']), tf.float32))
            ops['acc_spt'] = losses.accuracy(self.layers['spt_pred'], self.data['rel_spts'], name='acc_spt', ignore_bg=True)

class vggnet(sptnet):
    def __init__(self, data):
        super(vggnet, self).__init__(data)
        self.roi_scale = 1.0/16
        self._scope = 'vgg_16'

    def _net_conv(self, inputs):
        with tf.variable_scope(self._scope):
            # net = slim.conv2d(inputs, 64, [3, 3], scope='conv1_1')
            # net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
            # net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
            # net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
            # net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
            # net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
            # net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
            # net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
            # net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
            # net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool4')
            # net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
            # net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
            # net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            if self.stop_gradient: net = tf.stop_gradient(net, name='stop_gradient')
        return net

    def _net_roi_fc(self, inputs):
        with tf.variable_scope(self._scope):
            net = slim.fully_connected(inputs, 4096, scope='fc6')
            net = slim.dropout(net, self.keep_prob, scope='drop6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, self.keep_prob, scope='roi_fc_out')
            if self.stop_gradient: net = tf.stop_gradient(net, name='stop_gradient')
        return net

    def _net_rel_roi_fc(self, inputs):
        net = slim.fully_connected(inputs, 4096, scope='rel_fc6')
        net = slim.dropout(net, self.keep_prob, scope='rel_drop6')
        net = slim.fully_connected(net, 4096, scope='rel_fc7')
        net = slim.dropout(net, self.keep_prob, scope='rel_roi_fc_out')
        return net

    def get_variables_to_restore(self):
        variables = tf.global_variables(scope=self._scope)
        return variables

#
# class dual_graph_vrd(basenet):
#     def __init__(self, data):
#         basenet.__init__(self, data)
#
#         self.num_roi = data['num_roi']
#         self.num_rel = data['num_rel']
#         self.rel_rois = data['rel_rois']
#         self.iterable = True
#
#         self.edge_mask_inds = data['rel_mask_inds']
#         self.edge_segment_inds = data['rel_segment_inds']
#
#         self.edge_pair_mask_inds = data['rel_pair_mask_inds']
#         self.edge_pair_segment_inds = data['rel_pair_segment_inds']
#
#         # number of refine iterations
#         self.n_iter = data['n_iter']
#         self.relations = data['relations']
#
#         self.vert_state_dim = 512
#         self.edge_state_dim = 512
#
#     def setup(self):
#         self.layers = dict({'ims': self.ims, 'rois': self.rois, 'rel_rois': self.rel_rois})
#         self._vgg_conv()
#         self._vgg_fc()
#         self._union_rel_vgg_fc()
#         self._cells()
#         self._iterate()
#
#     def _cells(self):
#         """
#         construct RNN cells and states
#         """
#         # intiialize lstms
#         self.vert_rnn = tf.nn.rnn_cell.GRUCell(self.vert_state_dim, activation=tf.tanh)
#         self.edge_rnn = tf.nn.rnn_cell.GRUCell(self.edge_state_dim, activation=tf.tanh)
#
#         # lstm states
#         self.vert_state = self.vert_rnn.zero_state(self.num_roi, tf.float32)
#         self.edge_state = self.edge_rnn.zero_state(self.num_rel, tf.float32)
#
#     def _iterate(self):
#         (self.feed('vgg_out')
#              .fc(self.vert_state_dim, relu=False, name='vert_unary'))
#
#         (self.feed('rel_vgg_out')
#              .fc(self.edge_state_dim, relu=True, name='edge_unary'))
#
#         vert_unary = self.get_output('vert_unary')
#         edge_unary = self.get_output('edge_unary')
#         vert_factor = self._vert_rnn_forward(vert_unary, reuse=False)
#         edge_factor = self._edge_rnn_forward(edge_unary, reuse=False)
#
#         for i in xrange(self.n_iter):
#             reuse = i > 0
#             # compute edge states
#             edge_ctx = self._compute_edge_context(vert_factor, edge_factor, reuse=reuse)
#             edge_factor = self._edge_rnn_forward(edge_ctx, reuse=True)
#
#             # compute vert states
#             vert_ctx = self._compute_vert_context(edge_factor, vert_factor, reuse=reuse)
#             vert_factor = self._vert_rnn_forward(vert_ctx, reuse=True)
#             vert_in = vert_factor
#             edge_in = edge_factor
#
#             self._update_inference(vert_in, edge_in, i)
#
#     def _compute_edge_context_hard(self, vert_factor, reduction_mode='max'):
#         """
#         max or average message pooling
#         """
#         if reduction_mode=='max':
#             return tf.reduce_max(tf.gather(vert_factor, self.relations), [1])
#         elif reduction_mode=='mean':
#             return tf.reduce_mean(tf.gather(vert_factor, self.relations), [1])
#
#     def _compute_vert_context_hard(self, edge_factor, vert_factor, reduction_mode='max'):
#         """
#         max or average message pooling
#         """
#         edge_factor_gathered = utils.pad_and_gather(edge_factor, self.edge_mask_inds, None)
#
#         vert_ctx = utils.padded_segment_reduce(edge_factor_gathered, self.edge_segment_inds,
#                                                vert_factor.get_shape()[0], reduction_mode)
#
#         return vert_ctx
#
#     def _compute_edge_context_soft(self, vert_factor, edge_factor, reuse=False):
#         """
#         attention-based edge message pooling
#         """
#         vert_pairs = utils.gather_vec_pairs(vert_factor, self.relations)
#
#         sub_vert, obj_vert = tf.split(axis=1, num_or_size_splits=2, value=vert_pairs)
#         sub_vert_w_input = tf.concat(axis=1, values=[sub_vert, edge_factor])
#         obj_vert_w_input = tf.concat(axis=1, values=[obj_vert, edge_factor])
#
#
#         # compute compatibility scores
#         (self.feed(sub_vert_w_input)
#              .fc(1, relu=False, reuse=reuse, name='sub_vert_w_fc')
#              .sigmoid(name='sub_vert_score'))
#         (self.feed(obj_vert_w_input)
#              .fc(1, relu=False, reuse=reuse, name='obj_vert_w_fc')
#              .sigmoid(name='obj_vert_score'))
#
#         sub_vert_w = self.get_output('sub_vert_score')
#         obj_vert_w = self.get_output('obj_vert_score')
#
#         weighted_sub = tf.multiply(sub_vert, sub_vert_w)
#         weighted_obj = tf.multiply(obj_vert, obj_vert_w)
#         return weighted_sub + weighted_obj
#
#     def _compute_vert_context_soft(self, edge_factor, vert_factor, reuse=False):
#         """
#         attention-based vertex(node) message pooling
#         """
#
#         out_edge = utils.pad_and_gather(edge_factor, self.edge_pair_mask_inds[:,0])
#         in_edge = utils.pad_and_gather(edge_factor, self.edge_pair_mask_inds[:,1])
#         # gather correspounding vert factors
#         vert_factor_gathered = tf.gather(vert_factor, self.edge_pair_segment_inds)
#
#         # concat outgoing edges and ingoing edges with gathered vert_factors
#         out_edge_w_input = tf.concat(axis=1, values=[out_edge, vert_factor_gathered])
#         in_edge_w_input = tf.concat(axis=1, values=[in_edge, vert_factor_gathered])
#
#         # compute compatibility scores
#         (self.feed(out_edge_w_input)
#              .fc(1, relu=False, reuse=reuse, name='out_edge_w_fc')
#              .sigmoid(name='out_edge_score'))
#         (self.feed(in_edge_w_input)
#              .fc(1, relu=False, reuse=reuse, name='in_edge_w_fc')
#              .sigmoid(name='in_edge_score'))
#
#         out_edge_w = self.get_output('out_edge_score')
#         in_edge_w = self.get_output('in_edge_score')
#
#         # weight the edge factors with computed weigths
#         out_edge_weighted = tf.multiply(out_edge, out_edge_w)
#         in_edge_weighted = tf.multiply(in_edge, in_edge_w)
#
#
#         edge_sum = out_edge_weighted + in_edge_weighted
#         vert_ctx = tf.segment_sum(edge_sum, self.edge_pair_segment_inds)
#         return vert_ctx
#
#     def _vert_rnn_forward(self, vert_in, reuse=False):
#         with tf.variable_scope('vert_rnn'):
#             if reuse: tf.get_variable_scope().reuse_variables()
#             (vert_out, self.vert_state) = self.vert_rnn(vert_in, self.vert_state)
#         return vert_out
#
#     def _edge_rnn_forward(self, edge_in, reuse=False):
#         with tf.variable_scope('edge_rnn'):
#             if reuse: tf.get_variable_scope().reuse_variables()
#             (edge_out, self.edge_state) = self.edge_rnn(edge_in, self.edge_state)
#         return edge_out
#
#     def _update_inference(self, vert_factor, edge_factor, iter_i):
#         # make predictions
#         reuse = iter_i > 0  # reuse variables
#
#         iter_suffix = '_iter%i' % iter_i if iter_i < self.n_iter - 1 else ''
#         self._cls_pred(vert_factor, layer_suffix=iter_suffix, reuse=reuse)
#         self._bbox_pred(vert_factor, layer_suffix=iter_suffix, reuse=reuse)
#         self._rel_pred(edge_factor, layer_suffix=iter_suffix, reuse=reuse)
#
#     def losses(self):
#         return self._sg_losses()
#
#
# class vrd(basenet):
#     """
#     Baseline: the visual relation detection module proposed by
#     Lu et al.
#     """
#
#     def __init__(self, data):
#         basenet.__init__(self, data)
#         self.rel_rois = data['rel_rois']
#
#     def setup(self):
#         self.layers = dict({'ims': self.ims, 'rois': self.rois, 'rel_rois': self.rel_rois})
#         self._vgg_conv()
#         self._vgg_fc()
#         self._union_rel_vgg_fc()
#         self._cls_pred('vgg_out')
#         self._bbox_pred('vgg_out')
#         self._rel_pred('rel_vgg_out')
#
#     def losses(self):
#         return self._sg_losses()
#
#
# class dual_graph_vrd_maxpool(dual_graph_vrd):
#     """
#     Baseline: context-pooling by max pooling
#     """
#     def _compute_edge_context(self, vert_factor, edge_factor, reuse):
#         return self._compute_edge_context_hard(vert_factor, reduction_mode='max')
#
#     def _compute_vert_context(self, edge_factor, vert_factor, reuse):
#         return self._compute_vert_context_hard(edge_factor, vert_factor, reduction_mode='max')
#
#
# class dual_graph_vrd_avgpool(dual_graph_vrd):
#     """
#     Baseline: context-pooling by avg. pooling
#     """
#     def _compute_edge_context(self, vert_factor, edge_factor, reuse):
#         return self._compute_edge_context_hard(vert_factor, reduction_mode='mean')
#
#     def _compute_vert_context(self, edge_factor, vert_factor, reuse):
#         return self._compute_vert_context_hard(edge_factor, vert_factor, reduction_mode='mean')
#
#
# class dual_graph_vrd_final(dual_graph_vrd):
#     """
#     Our final model: context-pooling by attention
#     """
#     def _compute_edge_context(self, vert_factor, edge_factor, reuse):
#         return self._compute_edge_context_soft(vert_factor, edge_factor, reuse)
#
#     def _compute_vert_context(self, edge_factor, vert_factor, reuse):
#         return self._compute_vert_context_soft(edge_factor, vert_factor, reuse)
