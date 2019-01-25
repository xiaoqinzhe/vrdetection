import tensorflow as tf
from networks.models import *
from fast_rcnn.config import cfg
import networks.net_utils as utils
import tensorflow.contrib.slim as slim

class weightednet(basenet):
    def __init__(self, data):
        super(weightednet, self).__init__(data)
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox = False
        self.use_embedding = True
        self.use_spatial = True
        self.embedded_size = 256

    def _net(self):
        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane,
                             slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                            #weights_regularizer=weights_regularizer,
                            #biases_regularizer=biases_regularizer,
                            weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                            biases_initializer=tf.constant_initializer(0.0)
                            ):
            conv_net = self._net_conv(self.ims)
            self.layers['conv_out'] = conv_net
            roi_conv_out = self._net_roi_pooling([conv_net, self.rois], self.pooling_size, self.pooling_size,
                                                 name='roi_conv_out')
            rel_roi_conv_out = self._net_roi_pooling([conv_net, self.rel_rois], self.pooling_size,
                                                     self.pooling_size,
                                                     name='rel_roi_conv_out')
            roi_fc_out = self._net_roi_fc(roi_conv_out)
            self.rel_inx1, self.rel_inx2 = self._relation_indexes()

            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=weights_regularizer,
                                ):
                size=2048
                roi_fc_emb = slim.fully_connected(roi_fc_out, size)
                fc_sub = tf.gather(roi_fc_emb, self.rel_inx1)
                fc_obj = tf.gather(roi_fc_emb, self.rel_inx2)
                # conv 1 2
                conv_sub = tf.gather(roi_conv_out, self.rel_inx1)
                conv_obj = tf.gather(roi_conv_out, self.rel_inx2)
                net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
                net = slim.conv2d(net, 512, [3, 3])
                net = slim.conv2d(net, 512, [3, 3])
                net = slim.conv2d(net, 512, [3, 3])
                net = slim.flatten(net)
                net = slim.fully_connected(net, size)
                net = slim.dropout(net, keep_prob=self.keep_prob)
                net = slim.fully_connected(net, size)
                hhh=net = slim.dropout(net, keep_prob=self.keep_prob)
                vis_all = tf.concat([fc_sub, net, fc_obj], axis=1)
                vis = slim.fully_connected(vis_all, size)
                vis = slim.dropout(vis, keep_prob=self.keep_prob)
                vis = slim.fully_connected(vis, size)
                vis_feat = slim.dropout(vis, keep_prob=self.keep_prob)
                if self.use_embedding:
                    # class me
                    sub_emb, obj_emb = self._class_feature(self.rel_inx1, self.rel_inx2)
                    cls_emb = tf.concat([sub_emb, obj_emb], axis=1)
                    cls_proj = slim.fully_connected(cls_emb, 128)
                if self.use_spatial:
                    spt = self._spatial_feature(self.rel_inx1, self.rel_inx2)

                # weighted attention layer
                self.weighted_attention(conv_net, vis_feat, cls_proj=cls_proj, spt=spt)

                feat = tf.concat([vis_feat, cls_proj, spt], axis=1)

                with tf.variable_scope('rel_score'):
                    weight = tf.get_variable("weight", shape=[feat.shape.as_list()[1], self.num_predicates])
                rel_score = tf.matmul(feat, weight)
                self.layers['rel_score'] = rel_score
                self.layers['rel_prob'] = slim.softmax(rel_score, scope='rel_prob')

                # self._rel_pred(feat)

                att = tf.tile(self.layers['rel_weight_prob'], [1, self.num_predicates])
                self.layers['rel_weighted_prob'] = att * self.layers['rel_prob']
                self.layers['rel_pred'] = tf.argmax(self.layers['rel_prob'], axis=1, name='rel_pred')

    def weighted_attention(self, im_conv_out, vis_feat, cls_proj=None, spt=None):
        im_conv_out = self._net_roi_pooling([im_conv_out, self.data['rel_weight_rois']], 7, 7,
                                         name="im_roi_out")
        size=2048
        im_fc = self._net_conv_reshape(im_conv_out, "im_conv_reshape")
        im_fc = slim.fully_connected(im_fc, size)
        im_fc = slim.dropout(im_fc, keep_prob=self.keep_prob)
        im_fc = slim.fully_connected(im_fc, size)
        im_fc = slim.dropout(im_fc, keep_prob=self.keep_prob)

        im_fc_gather = tf.gather(im_fc, tf.cast(self.rel_rois[:, 0], tf.int32))
        # vis_feat = slim.fully_connected(vis_feat, size)
        # vis_feat = slim.dropout(vis_feat, self.keep_prob)
        # vis_feat = slim.fully_connected(vis_feat, size)
        ctx = tf.concat([im_fc_gather, vis_feat], axis=1)
        ctx = slim.fully_connected(ctx, size)
        ctx = slim.dropout(ctx, keep_prob=self.keep_prob)
        # ctx = slim.fully_connected(ctx, size)
        # ctx = slim.dropout(ctx, keep_prob=self.keep_prob)
        if cls_proj is not None:
            cls_proj = slim.fully_connected(cls_proj, 128)
            ctx = tf.concat([ctx, cls_proj], axis=1)
        if spt is not None:
            ctx = tf.concat([ctx, spt], axis=1)
        a = slim.fully_connected(ctx, 1, activation_fn=None, scope="rel_weights")
        self.layers['rel_weight_score'] = a
        self.layers['rel_weight_prob'] = tf.sigmoid(a)

    def metrics(self, ops={}):
        super(weightednet, self).metrics(ops)
        prob = tf.squeeze(self.layers['rel_weight_prob'])
        thresh = tf.ones_like(prob, tf.float32) * 0.5
        pred = tf.cast(tf.greater_equal(prob, thresh), tf.int32)
        true_pred = tf.cast(tf.equal(pred, self.data['rel_weight_labels']), tf.float32)
        ops['acc_weight'] = tf.reduce_mean(true_pred)
        num_pos = tf.reduce_sum(tf.cast(self.data['rel_weight_labels'], tf.float32))
        pos_inds = tf.where(tf.cast(self.data['rel_weight_labels'], tf.bool))
        pos_pred = tf.reduce_sum(tf.gather(true_pred, pos_inds))
        ops['rec_weight'] =  pos_pred/num_pos

    # Losses
    def losses(self):
        losses = super(weightednet, self).losses()
        weight_loss = tf.losses.sigmoid_cross_entropy(tf.expand_dims(self.data['rel_weight_labels'], axis=1), self.layers['rel_weight_score'])
        # weight_loss = tf.reduce_mean(tf.square(self.layers['rel_weight_pred']-self.data['rel_weight_labels']))
        # weight_loss = tf.reduce_mean(tf.abs(self.layers['rel_weights'] - self.data['rel_weights']))
        print(weight_loss)
        losses['loss_weight'] = tf.reduce_mean(weight_loss)
        losses['loss_total'] = tf.add(losses['loss_total'], losses['loss_weight'])
        return losses

class ranknet(basenet):
    def __init__(self, data):
        super(ranknet, self).__init__(data)
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox = False
        self.use_embedding = True
        self.use_spatial = True
        self.embedded_size = 128
        self.if_pred_weight = True
        self.rel_weight_labels = self.data['rel_weight_labels']
        if self.is_training:
            self.rel_triple_inds = self.data['rel_triple_inds']
            self.rel_triple_labels = self.data['rel_triple_labels']

    def _net(self):
        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane,
                             slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                            #weights_regularizer=weights_regularizer,
                            biases_regularizer=biases_regularizer,
                            weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                            biases_initializer=tf.constant_initializer(0.0)
                            ):
            conv_net = self._net_conv(self.ims)
            self.layers['conv_out'] = conv_net
            roi_conv_out = self._net_roi_pooling([conv_net, self.rois], self.pooling_size, self.pooling_size,
                                                 name='roi_conv_out')
            rel_roi_conv_out = self._net_roi_pooling([conv_net, self.rel_rois], self.pooling_size,
                                                     self.pooling_size,
                                                     name='rel_roi_conv_out')
            roi_fc_out = self._net_roi_fc(roi_conv_out)
            self.rel_inx1, self.rel_inx2 = self._relation_indexes()
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=weights_regularizer,
                    ):
                size=2048
                roi_fc_emb = slim.fully_connected(roi_fc_out, size)
                fc_sub = tf.gather(roi_fc_emb, self.rel_inx1)
                fc_obj = tf.gather(roi_fc_emb, self.rel_inx2)
                # conv 1 2
                conv_sub = tf.gather(roi_conv_out, self.rel_inx1)
                conv_obj = tf.gather(roi_conv_out, self.rel_inx2)
                net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
                net = slim.conv2d(net, 512, [3, 3])
                net = slim.conv2d(net, 512, [3, 3])
                net = slim.conv2d(net, 512, [3, 3])
                net = slim.flatten(net)
                net = slim.fully_connected(net, size)
                net = slim.dropout(net, keep_prob=self.keep_prob)
                net = slim.fully_connected(net, size)
                net = slim.dropout(net, keep_prob=self.keep_prob)
                vis_all = tf.concat([fc_sub, net, fc_obj], axis=1)
                vis = slim.fully_connected(vis_all, size)
                vis = slim.dropout(vis, keep_prob=self.keep_prob)
                vis = slim.fully_connected(vis, size)
                vis_feat = slim.dropout(vis, keep_prob=self.keep_prob)
                if self.use_embedding:
                    # class me
                    sub_emb, obj_emb = self._class_feature(self.rel_inx1, self.rel_inx2)
                    cls_emb = tf.concat([sub_emb, obj_emb], axis=1)
                    cls_proj = slim.fully_connected(cls_emb, 128)
                if self.use_spatial:
                    spt = self._spatial_feature(self.rel_inx1, self.rel_inx2)

                feat=vis_feat
                self.weighted_attention(conv_net, feat, cls_proj=cls_proj, spt=spt)

                # triple ranking net
                if self.is_training:
                    self.triple_net()

                feat = tf.concat([vis_feat, cls_proj, spt], axis=1)

                # self._rel_pred(feat)

                with tf.variable_scope('rel_score'):
                    weight = tf.get_variable("weight", shape=[feat.shape.as_list()[1], self.num_predicates])
                rel_score = tf.matmul(feat, weight)
                self.layers['rel_score'] = rel_score
                self.layers['rel_prob'] = slim.softmax(rel_score, scope='rel_prob')
                att = tf.tile(self.layers['rel_weight_prob'], [1, self.num_predicates])
                self.layers['rel_weighted_prob'] = att * self.layers['rel_prob']
                self.layers['rel_pred'] = tf.argmax(self.layers['rel_prob'], axis=1, name='rel_pred')

    def weighted_attention(self, im_conv_out, vis_feat, cls_proj=None, spt=None):
        im_conv_out = self._net_roi_pooling([im_conv_out, self.data['rel_weight_rois']], 7, 7,
                                            name="im_roi_out")
        size = 2048
        im_fc = self._net_conv_reshape(im_conv_out, "im_conv_reshape")
        im_fc = slim.fully_connected(im_fc, size)
        im_fc = slim.dropout(im_fc, keep_prob=self.keep_prob)
        im_fc = slim.fully_connected(im_fc, size)
        im_fc = slim.dropout(im_fc, keep_prob=self.keep_prob)

        im_fc_gather = tf.gather(im_fc, tf.cast(self.rel_rois[:, 0], tf.int32))
        # vis_feat = slim.fully_connected(vis_feat, 1024)
        # vis_feat = slim.dropout(vis_feat, self.keep_prob)
        # vis_feat = slim.fully_connected(vis_feat, 1024)
        ctx = tf.concat([im_fc_gather, vis_feat], axis=1)
        ctx = slim.fully_connected(ctx, size)
        ctx = slim.dropout(ctx, keep_prob=self.keep_prob)
        if cls_proj is not None:
            cls_proj = slim.fully_connected(cls_proj, 128)
            ctx = tf.concat([ctx, cls_proj], axis=1)
        if spt is not None:
            ctx = tf.concat([ctx, spt], axis=1)
        a = slim.fully_connected(ctx, 1, activation_fn=None, scope="rel_weights")
        self.layers['rel_weight_score'] = a
        self.layers['rel_weight_prob'] = tf.sigmoid(a)

    def triple_net(self):
        rel_weights = self.layers['rel_weight_prob']
        # rel_weights = slim.fully_connected(self.tempf, 1, activation_fn=None)
        pair1 = tf.gather(rel_weights, self.rel_triple_inds[:, 0])
        pair2 = tf.gather(rel_weights, self.rel_triple_inds[:, 1])
        self.layers['rel_triple_score'] = pair1 - pair2
        self.layers['rel_triple_prob'] = tf.sigmoid(self.layers['rel_triple_score'])

    def metrics(self, ops={}):
        super(basenet, self).metrics(ops)
        prob = tf.squeeze(self.layers['rel_weight_prob'])
        thresh = tf.ones_like(prob, tf.float32) * 0.5
        pred = tf.cast(tf.greater_equal(prob, thresh), tf.int32)
        true_pred = tf.cast(tf.equal(pred, self.data['rel_weight_labels']), tf.float32)
        ops['acc_weight'] = tf.reduce_mean(true_pred)
        num_pos = tf.reduce_sum(tf.cast(self.data['rel_weight_labels'], tf.float32))
        pos_inds = tf.where(tf.cast(self.data['rel_weight_labels'], tf.bool))
        pos_pred = tf.reduce_sum(tf.gather(true_pred, pos_inds))
        ops['rec_weight'] =  pos_pred/num_pos
        if self.is_training:
            # triple accuracy
            thresh = tf.ones_like(self.rel_triple_labels, tf.float32) * 0.5
            pred = tf.cast(tf.greater_equal(self.layers['rel_triple_prob'], thresh), tf.int32)
            true_pred = tf.cast(tf.equal(pred, self.rel_triple_labels), tf.float32)
            ops['acc_triple'] = tf.reduce_mean(true_pred)

    # Losses
    def losses(self):
        losses = super(basenet, self).losses()
        weight_loss = tf.losses.sigmoid_cross_entropy(tf.expand_dims(self.rel_weight_labels, axis=1), self.layers['rel_weight_score'])
        losses['loss_weight'] = tf.reduce_mean(weight_loss)
        losses['loss_total'] = tf.add(losses['loss_total'], losses['loss_weight'])
        if self.is_training:
            losses['loss_triple'] = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.rel_triple_labels, self.layers['rel_triple_score']))
            # losses['loss_triple'] = tf.reduce_mean(tf.abs(self.layers['rel_triple_score']-tf.cast(self.rel_triple_labels, tf.float32)))
            losses['loss_total'] = tf.add(losses['loss_total'], losses['loss_triple'])
        return losses

class weightednet2(weightednet):
    def __init__(self, data):
        super(weightednet2, self).__init__(data)
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox = False
        self.use_embedding = True
        self.use_spatial = True
        self.embedded_size = 256
        self.obj_matrix = self.data['obj_matrix']
        self.rel_matrix = self.data['rel_matrix']

    def _net(self):
        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane,
                             slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                            # weights_regularizer=weights_regularizer,
                            biases_regularizer=biases_regularizer,
                            weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                            biases_initializer=tf.constant_initializer(0.0)
                            ):
            conv_net = self._net_conv(self.ims)
            self.layers['conv_out'] = conv_net
            roi_conv_out = self._net_roi_pooling([conv_net, self.rois], self.pooling_size, self.pooling_size,
                                                 name='roi_conv_out')
            rel_roi_conv_out = self._net_roi_pooling([conv_net, self.rel_rois], self.pooling_size,
                                                     self.pooling_size,
                                                     name='rel_roi_conv_out')
            roi_fc_out = self._net_roi_fc(roi_conv_out)
            self.rel_inx1, self.rel_inx2 = self._relation_indexes()
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=weights_regularizer,
                                ):
                size = 2048
                roi_fc_emb = slim.fully_connected(roi_fc_out, size)
                fc_sub = tf.gather(roi_fc_emb, self.rel_inx1)
                fc_obj = tf.gather(roi_fc_emb, self.rel_inx2)
                # conv 1 2
                conv_sub = tf.gather(roi_conv_out, self.rel_inx1)
                conv_obj = tf.gather(roi_conv_out, self.rel_inx2)
                net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
                net = slim.conv2d(net, 512, [3, 3])
                net = slim.conv2d(net, 512, [3, 3])
                net = slim.conv2d(net, 512, [3, 3])
                net = slim.flatten(net)
                net = slim.fully_connected(net, size)
                net = slim.dropout(net, keep_prob=self.keep_prob)
                net = slim.fully_connected(net, size)
                net = slim.dropout(net, keep_prob=self.keep_prob)
                vis_all = tf.concat([fc_sub, net, fc_obj], axis=1)
                vis = slim.fully_connected(vis_all, size)
                vis = slim.dropout(vis, keep_prob=self.keep_prob)
                vis = slim.fully_connected(vis, size)
                vis_d = slim.dropout(vis, keep_prob=self.keep_prob)
                if self.use_embedding:
                    # class me
                    sub_emb, obj_emb = self._class_feature(self.rel_inx1, self.rel_inx2)
                    cls_emb = tf.concat([sub_emb, obj_emb], axis=1)
                    cls_proj = slim.fully_connected(cls_emb, 128)
                if self.use_spatial:
                    spt = self._spatial_feature(self.rel_inx1, self.rel_inx2)

                # ......
                vis_feat = vis_d

                # gcn
                # vis_feat = self.gcn(vis, self.rel_matrix, "gcn1", next_dim=2048)
                # vis_feat = self.gcn(vis_feat, self.rel_matrix, "gcn11", next_dim=2048)
                # vis_feat = self.gcn(vis_feat, self.rel_matrix, "gcn2", next_dim=1024)
                # vis_feat = self.gcn(vis_feat, self.rel_matrix, "gcn22", next_dim=1024)
                # vis_feat = self.gcn(vis_feat, self.rel_matrix, "gcn3", next_dim=512)
                # vis_feat = self.gcn(vis_feat, self.rel_matrix, "gcn33", next_dim=512)
                # vis_feat = slim.dropout(vis_feat, keep_prob=self.keep_prob)
                # vis_feat = tf.concat([vis_d, vis_feat], axis=1)

                # weighted attention layer
                self.weighted_attention(conv_net, vis_feat, cls_proj=cls_proj, spt=spt)

                feat = tf.concat([vis_feat, cls_proj, spt], axis=1)

                with tf.variable_scope('rel_score'):
                    weight = tf.get_variable("weight", shape=[feat.shape.as_list()[1], self.num_predicates])
                rel_score = tf.matmul(feat, weight)
                self.layers['rel_score'] = rel_score
                self.layers['rel_prob'] = slim.softmax(rel_score, scope='rel_prob')
                att = tf.tile(self.layers['rel_weight_prob'], [1, self.num_predicates])
                self.layers['rel_weighted_prob'] = att * self.layers['rel_prob']
                self.layers['rel_pred'] = tf.argmax(self.layers['rel_prob'], axis=1, name='rel_pred')

    def weighted_attention(self, im_conv_out, vis_feat, cls_proj=None, spt=None):
        im_conv_out = self._net_roi_pooling([im_conv_out, self.data['rel_weight_rois']], 7, 7,
                                            name="im_roi_out")
        size = 2048
        im_fc = self._net_conv_reshape(im_conv_out, "im_conv_reshape")
        im_fc = slim.fully_connected(im_fc, size)
        im_fc = slim.dropout(im_fc, keep_prob=self.keep_prob)
        im_fc = slim.fully_connected(im_fc, size)
        im_fc = slim.dropout(im_fc, keep_prob=self.keep_prob)

        im_fc_gather = tf.gather(im_fc, tf.cast(self.rel_rois[:, 0], tf.int32))
        # vis_feat = slim.fully_connected(vis_feat, 1024)
        # vis_feat = slim.dropout(vis_feat, self.keep_prob)
        # vis_feat = slim.fully_connected(vis_feat, 1024)
        ctx = tf.concat([im_fc_gather, vis_feat], axis=1)
        ctx = slim.fully_connected(ctx, size)
        ctx = slim.dropout(ctx, keep_prob=self.keep_prob)
        if cls_proj is not None:
            cls_proj = slim.fully_connected(cls_proj, 128)
            ctx = tf.concat([ctx, cls_proj], axis=1)
        if spt is not None:
            ctx = tf.concat([ctx, spt], axis=1)

        # propagation
        v = self.gcn(ctx, self.rel_matrix, "gcn1", next_dim=2048)
        # vis_feat = self.gcn(vis_feat, self.rel_matrix, "gcn11", next_dim=512)
        v = self.gcn(v, self.rel_matrix, "gcn2", next_dim=1024)
        # vis_feat = self.gcn(vis_feat, self.rel_matrix, "gcn22", next_dim=256)
        v = self.gcn(v, self.rel_matrix, "gcn3", next_dim=512)
        # vis_feat = self.gcn(vis_feat, self.rel_matrix, "gcn33", next_dim=128)
        ctx = tf.concat([ctx, v], axis=1)

        a = slim.fully_connected(ctx, 1, activation_fn=None, scope="rel_weights")
        self.layers['rel_weight_score'] = a
        self.layers['rel_weight_prob'] = tf.sigmoid(a)

    def gcn(self, inputs, matrix, name, next_dim=None, reuse=False, activation=tf.sigmoid):
        with tf.variable_scope(name, reuse=reuse):
            if next_dim is None: next_dim = inputs.shape.as_list()[1]
            weight = tf.get_variable("weight", shape=[inputs.shape.as_list()[1], next_dim])
            net = tf.matmul(tf.matmul(matrix, inputs), weight)
            net = activation(net)
        return net