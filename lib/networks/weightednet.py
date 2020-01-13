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
        self.embedded_size = 256
        self.fuse_size = 512
        self.if_fuse = False
        print(self.use_vis, self.use_embedding, self.use_spatial)

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
            # roi_conv_out = self._net_crop_pooling([conv_net, self.rois], self.pooling_size,
            #                                      name='roi_conv_out')
            # rel_roi_conv_out = self._net_crop_pooling([conv_net, self.rel_rois], self.pooling_size,
            #                                          name='rel_roi_conv_out')
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

                # self.use_embedding = False
                # self.use_spatial = False
                if self.use_embedding:
                    # class me
                    sub_emb, obj_emb = self._class_feature(self.rel_inx1, self.rel_inx2)
                    cls_emb = tf.concat([sub_emb, obj_emb], axis=1)
                    cls_proj = slim.fully_connected(cls_emb, 128)
                else:   cls_proj = None
                if self.use_spatial:
                    spt = self._spatial_feature(self.rel_inx1, self.rel_inx2)
                else:   spt = None

                # weighted attention layer
                if not self.use_vis:  vis_feat=None
                self.weighted_attention(conv_net, vis_feat, cls_proj=cls_proj, spt=spt)
                # self.weighted_attention(conv_net, vis_feat)

                if self.use_vis:
                    # if self.if_fuse: vis_feat = slim.fully_connected(vis_feat, self.fuse_size)
                    feat = vis_feat
                else:
                    feat = None
                if self.use_embedding:
                    # if self.if_fuse: cls_proj = slim.fully_connected(cls_proj, self.fuse_size)
                    if feat is None:  feat = cls_proj
                    else:  feat = tf.concat([feat, cls_proj], axis=1)
                if self.use_spatial:
                    # if self.if_fuse: spt = slim.fully_connected(spt, self.fuse_size)
                    if feat is None:  feat = spt
                    else: feat = tf.concat([feat, spt], axis=1)
                # feat = tf.concat([vis_feat, cls_proj, spt], axis=1)

                with tf.variable_scope('rel_score'):
                    weight = tf.get_variable("weight", shape=[feat.shape.as_list()[1], self.num_predicates])
                rel_score = tf.matmul(feat, weight)
                self.layers['rel_score'] = rel_score
                self.layers['rel_prob'] = slim.softmax(rel_score, scope='rel_prob')

                # self._rel_pred(feat)

                att = tf.tile(self.layers['rel_weight_prob'], [1, self.num_predicates])
                self.layers['rel_weighted_prob'] = att
                self.layers['rel_pred'] = tf.argmax(self.layers['rel_prob'], axis=1, name='rel_pred')

    def weighted_attention(self, im_conv_out, vis_feat, cls_proj=None, spt=None):
        if vis_feat is not None:
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
        else: ctx=None
        if cls_proj is not None:
            cls_proj = slim.fully_connected(cls_proj, 128)
            if ctx is not None: ctx = tf.concat([ctx, cls_proj], axis=1)
            else: ctx=cls_proj
        if spt is not None:
            if ctx is not None: ctx = tf.concat([ctx, spt], axis=1)
            else: ctx=spt
        assert ctx is not None
        a = slim.fully_connected(ctx, 1, activation_fn=None, scope="rel_weights")
        self.layers['rel_weight_score'] = a
        b=tf.exp(tf.squeeze(a))
        b=b/tf.reduce_sum(b)
        self.layers['rel_weight_soft'] = tf.expand_dims(b, -1)
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
        #print(weight_loss)
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

class simplenet(basenet):
    def __init__(self, data):
        super(simplenet, self).__init__(data)
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox = False
        self.embedded_size = 256
        self.fuse_size = 512
        self.rel_prior = self.data['prior'] if 'prior' in self.data else None
        self.if_fuse = True
        print(self.use_vis, self.use_embedding, self.use_spatial)

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
            # roi_conv_out = self._net_crop_pooling([conv_net, self.rois], self.pooling_size,
            #                                      name='roi_conv_out')
            # rel_roi_conv_out = self._net_crop_pooling([conv_net, self.rel_rois], self.pooling_size,
            #                                          name='rel_roi_conv_out')
            roi_fc_out = self._net_roi_fc(roi_conv_out)
            rel_roi_fc_out = self._net_roi_fc(rel_roi_conv_out, reuse=True)
            self.rel_inx1, self.rel_inx2 = self._relation_indexes()

            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=weights_regularizer,
                                ):
                size = 2048
                roi_fc_emb = slim.fully_connected(roi_fc_out, size)
                fc_sub = tf.gather(roi_fc_emb, self.rel_inx1)
                fc_obj = tf.gather(roi_fc_emb, self.rel_inx2)
                fc_pred = slim.fully_connected(rel_roi_fc_out, size)
                vis_feat = slim.fully_connected(tf.concat([fc_sub, fc_pred, fc_obj], axis=1), size)
                vis_feat = slim.dropout(vis_feat, keep_prob=self.keep_prob)
                vis_feat = slim.fully_connected(vis_feat, size)
                vis_feat = slim.dropout(vis_feat, keep_prob=self.keep_prob)

                emb_size = 2048

                if self.use_embedding:
                    # class me
                    sub_emb, obj_emb = self._class_feature(self.rel_inx1, self.rel_inx2)
                    # cls_sub = slim.fully_connected(sub_emb, emb_size, scope="cls_proj")
                    # cls_obj = slim.fully_connected(sub_emb, emb_size, scope="cls_proj", reuse=True)
                    # cls_proj = slim.fully_connected(tf.concat([cls_sub, cls_obj, prior], 1), emb_size)
                    # cls_proj = slim.fully_connected(tf.concat([cls_sub, cls_obj], 1), emb_size)
                    cls_emb = tf.concat([sub_emb, obj_emb], axis=1)
                    cls_proj = slim.fully_connected(cls_emb, emb_size)
                else:
                    cls_proj = None
                if self.use_spatial:
                    spt = self._spatial_feature(self.rel_inx1, self.rel_inx2)
                    spt = slim.fully_connected(spt, emb_size)
                else:
                    spt = None



                # weighted attention layer
                if not self.use_vis:  vis_feat = None
                self.weighted_attention(conv_net, vis_feat, cls_proj=cls_proj, spt=spt)
                # self.weighted_attention(conv_net, vis_feat)

                # attention vision
                fc_sub_proj = slim.fully_connected(fc_sub, emb_size, scope="proj_obj_fc")
                fc_obj_proj = slim.fully_connected(fc_obj, emb_size, scope="proj_obj_fc", reuse=True)
                vis_ctx = slim.fully_connected(tf.concat([fc_sub_proj, fc_obj_proj, cls_proj, spt], axis=1), emb_size)
                vis_feat_att = self.local_attention(vis_feat, rel_roi_conv_out)

                if self.use_vis:
                    vis_feat = slim.fully_connected(vis_feat, size)
                    # vis_feat = vis_feat_att
                    feat = vis_feat
                else:
                    feat = None
                if self.use_embedding:
                    cls_proj = slim.fully_connected(cls_proj, size)
                    if feat is None:
                        feat = cls_proj
                    else:
                        feat = tf.concat([feat, cls_proj], axis=1)
                if self.use_spatial:
                    spt = slim.fully_connected(spt, size)
                    if feat is None:
                        feat = spt
                    else:
                        feat = tf.concat([feat, spt], axis=1)

                feat = slim.fully_connected(feat, size)
                # prior = slim.fully_connected(self.rel_prior, emb_size)
                # feat = tf.concat([feat, prior], axis=1)

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
        if vis_feat is not None:
            im_conv_out = self._net_roi_pooling([im_conv_out, self.data['rel_weight_rois']], 7, 7,
                                             name="im_roi_out")
            size=2048
            im_fc = self._net_conv_reshape(im_conv_out, "im_conv_reshape")
            im_fc = slim.fully_connected(im_fc, size)
            im_fc = slim.dropout(im_fc, keep_prob=self.keep_prob)
            im_fc = slim.fully_connected(im_fc, size)
            im_fc = slim.dropout(im_fc, keep_prob=self.keep_prob)

            im_fc_gather = tf.gather(im_fc, tf.cast(self.rel_rois[:, 0], tf.int32))
            vis_feat = slim.fully_connected(vis_feat, size)
            # vis_feat = slim.dropout(vis_feat, self.keep_prob)
            # vis_feat = slim.fully_connected(vis_feat, size)
            ctx = tf.concat([im_fc_gather, vis_feat], axis=1)
            ctx = slim.fully_connected(ctx, size)
            ctx = slim.dropout(ctx, keep_prob=self.keep_prob)
            # ctx = slim.fully_connected(ctx, size)
            # ctx = slim.dropout(ctx, keep_prob=self.keep_prob)
        else: ctx=None
        if cls_proj is not None:
            cls_proj = slim.fully_connected(cls_proj, 2048)
            if ctx is not None: ctx = tf.concat([ctx, cls_proj], axis=1)
            else: ctx=cls_proj
        if spt is not None:
            spt = slim.fully_connected(spt, 2048)
            if ctx is not None: ctx = tf.concat([ctx, spt], axis=1)
            else: ctx=spt
        assert ctx is not None
        a = slim.fully_connected(ctx, 1, activation_fn=None, scope="rel_weights")
        self.layers['rel_weight_score'] = a
        b=tf.exp(tf.squeeze(a))
        b=b/tf.reduce_sum(b)
        self.layers['rel_weight_soft'] = tf.expand_dims(b, -1)
        self.layers['rel_weight_prob'] = tf.sigmoid(a)

    def metrics(self, ops={}):
        super(simplenet, self).metrics(ops)
        prob = tf.squeeze(self.layers['rel_weight_prob'])
        thresh = tf.ones_like(prob, tf.float32) * 0.5
        pred = tf.cast(tf.greater_equal(prob, thresh), tf.int32)
        true_pred = tf.cast(tf.equal(pred, self.data['rel_weight_labels']), tf.float32)
        ops['acc_weight'] = tf.reduce_mean(true_pred)
        num_pos = tf.reduce_sum(tf.cast(self.data['rel_weight_labels'], tf.float32))
        pos_inds = tf.where(tf.cast(self.data['rel_weight_labels'], tf.bool))
        pos_pred = tf.reduce_sum(tf.gather(true_pred, pos_inds))
        ops['rec_weight'] =  pos_pred/num_pos

    def lk_loss(self, labels, predictions, prior):
        ont_hot_labels = tf.one_hot(labels, self.num_predicates)
        sl = 0.0
        alpha = 0.8
        smooth_labels = alpha*(ont_hot_labels * (1 - sl) + sl/self.num_predicates) + (1-alpha)*prior
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=smooth_labels, logits=predictions)
        if True:  # do not penalize background class
            loss_mask = tf.cast(tf.greater(labels, 0), tf.float32)
            loss = tf.multiply(loss, loss_mask)
            loss = tf.reduce_sum(loss)/(tf.reduce_sum(loss_mask)+1)
        else:
            loss = tf.reduce_sum(loss)
        return loss

    # Losses
    def losses(self):
        losses = super(simplenet, self).losses()
        weight_loss = tf.losses.sigmoid_cross_entropy(tf.expand_dims(self.data['rel_weight_labels'], axis=1), self.layers['rel_weight_score'])
        # weight_loss = tf.reduce_mean(tf.square(self.layers['rel_weight_pred']-self.data['rel_weight_labels']))
        # weight_loss = tf.reduce_mean(tf.abs(self.layers['rel_weights'] - self.data['rel_weights']))
        #print(weight_loss)
        losses['loss_weight'] = tf.reduce_mean(weight_loss)
        alphas = [1, 1]
        losses['loss_total'] = tf.add(losses['loss_reg'], tf.add(alphas[0]*losses['loss_rel'], alphas[1]*losses['loss_weight']))
        # prior loss
        use_prior = False
        if use_prior:
            beta = 0.8
            print(self.rel_prior.shape)
            losses['loss_rel'] = self.lk_loss(self.predicates, self.layers['rel_score'], self.rel_prior)
            losses['loss_total'] = 0.2 * losses['loss_reg'] + alphas[0] * losses['loss_rel'] + alphas[1] * losses['loss_weight']
            # losses['loss_prior'] = tf.losses.sigmoid_cross_entropy(self.rel_prior, self.layers['rel_score'])
            # losses['loss_total'] = 0.2*losses['loss_reg'] + alpha*(beta*losses['loss_rel'] + (1-beta)*losses['loss_prior']) + (1-alpha)*losses['loss_weight']
        return losses


class weightednet3(basenet):
    def __init__(self, data):
        super(weightednet3, self).__init__(data)
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox = False
        self.embedded_size = 256
        self.fuse_size = 512
        self.if_fuse = False
        print(self.use_vis, self.use_embedding, self.use_spatial)

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
            # roi_conv_out = self._net_crop_pooling([conv_net, self.rois], self.pooling_size,
            #                                      name='roi_conv_out')
            # rel_roi_conv_out = self._net_crop_pooling([conv_net, self.rel_rois], self.pooling_size,
            #                                          name='rel_roi_conv_out')
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

                # self.use_embedding = False
                # self.use_spatial = False
                if self.use_embedding:
                    # class me
                    sub_emb, obj_emb = self._class_feature(self.rel_inx1, self.rel_inx2)
                    cls_emb = tf.concat([sub_emb, obj_emb], axis=1)
                    cls_proj = slim.fully_connected(cls_emb, 128)
                else:   cls_proj = None
                if self.use_spatial:
                    spt = self._spatial_feature(self.rel_inx1, self.rel_inx2)
                else:   spt = None

                if self.use_vis:
                    # if self.if_fuse: vis_feat = slim.fully_connected(vis_feat, self.fuse_size)
                    feat = vis_feat
                else:
                    feat = None
                if self.use_embedding:
                    # if self.if_fuse: cls_proj = slim.fully_connected(cls_proj, self.fuse_size)
                    if feat is None:  feat = cls_proj
                    else:  feat = tf.concat([feat, cls_proj], axis=1)
                if self.use_spatial:
                    # if self.if_fuse: spt = slim.fully_connected(spt, self.fuse_size)
                    if feat is None:  feat = spt
                    else: feat = tf.concat([feat, spt], axis=1)
                # feat = tf.concat([vis_feat, cls_proj, spt], axis=1)

                with tf.variable_scope('rel_score'):
                    weight = tf.get_variable("weight", shape=[feat.shape.as_list()[1], self.num_predicates])
                rel_score = tf.matmul(feat, weight)
                self.layers['rel_score'] = rel_score
                self.layers['rel_prob'] = slim.softmax(rel_score, scope='rel_prob')
                self.layers['rel_pred'] = tf.argmax(self.layers['rel_prob'], axis=1, name='rel_pred')

                # self._rel_pred(feat)

                # weighted attention layer
                if not self.use_vis:  vis_feat = None
                self.weighted_attention(conv_net, vis_feat, cls_proj=cls_proj, spt=spt)
                self.roi_attention(conv_net, roi_fc_out)

                fg_prob = self.layers["fg_prob"]
                fg_prob_sub = tf.gather(fg_prob, self.rel_inx1)
                fg_prob_obj = tf.gather(fg_prob, self.rel_inx2)
                self.layers['rel_weight_prob'] = self.layers['rel_weight_prob'] * fg_prob_sub * fg_prob_obj
                att = tf.tile(self.layers['rel_weight_prob'], [1, self.num_predicates])
                self.layers['rel_weighted_prob'] = att


    def weighted_attention(self, im_conv_out, vis_feat, cls_proj=None, spt=None):
        if vis_feat is not None:
            im_conv_out = self._net_roi_pooling([im_conv_out, self.data['rel_weight_rois']], 7, 7,
                                             name="im_roi_out")
            size=2048
            im_fc = self._net_conv_reshape(im_conv_out, "im_conv_reshape")
            im_fc = slim.fully_connected(im_fc, size)
            im_fc = slim.dropout(im_fc, keep_prob=self.keep_prob)
            im_fc = slim.fully_connected(im_fc, size)
            self.layers['im_fc'] = im_fc = slim.dropout(im_fc, keep_prob=self.keep_prob)

            im_fc_gather = tf.gather(im_fc, tf.cast(self.rel_rois[:, 0], tf.int32))
            # vis_feat = slim.fully_connected(vis_feat, size)
            # vis_feat = slim.dropout(vis_feat, self.keep_prob)
            # vis_feat = slim.fully_connected(vis_feat, size)
            ctx = tf.concat([im_fc_gather, vis_feat], axis=1)
            ctx = slim.fully_connected(ctx, size)
            ctx = slim.dropout(ctx, keep_prob=self.keep_prob)
            # ctx = slim.fully_connected(ctx, size)
            # ctx = slim.dropout(ctx, keep_prob=self.keep_prob)
        else: ctx=None
        if cls_proj is not None:
            cls_proj = slim.fully_connected(cls_proj, 128)
            if ctx is not None: ctx = tf.concat([ctx, cls_proj], axis=1)
            else: ctx=cls_proj
        if spt is not None:
            if ctx is not None: ctx = tf.concat([ctx, spt], axis=1)
            else: ctx=spt
        assert ctx is not None
        a = slim.fully_connected(ctx, 1, activation_fn=None, scope="rel_weights")
        self.layers['rel_weight_score'] = a
        b=tf.exp(tf.squeeze(a))
        b=b/tf.reduce_sum(b)
        self.layers['rel_weight_soft'] = tf.expand_dims(b, -1)
        self.layers['rel_weight_prob'] = tf.sigmoid(a)

    def roi_attention(self, im_conv_out, vis_feat):
        size = 2048
        im_fc_gather = tf.gather(slim.fully_connected(self.layers['im_fc'], size), tf.cast(self.rois[:, 0], tf.int32))
        vis_feat = slim.fully_connected(vis_feat, size)
        # vis_feat = slim.dropout(vis_feat, self.keep_prob)
        # vis_feat = slim.fully_connected(vis_feat, size)
        ctx = tf.concat([im_fc_gather, vis_feat], axis=1)
        #
        # ctx = vis_feat
        ctx = slim.fully_connected(ctx, size)
        # ctx = slim.dropout(ctx, keep_prob=self.keep_prob)
        # ctx = slim.fully_connected(ctx, size)
        # ctx = slim.dropout(ctx, keep_prob=self.keep_prob)
        assert ctx is not None
        a = slim.fully_connected(ctx, 1, activation_fn=None, scope="fg_weights")
        self.layers['fg_score'] = a
        # b=tf.exp(tf.squeeze(a))
        # b=b/tf.reduce_sum(b)
        # self.layers['fg_soft'] = tf.expand_dims(b, -1)
        self.layers['fg_prob'] = tf.sigmoid(a)

    def get_acc_rec(self, probs, labels):
        prob = tf.squeeze(probs)
        thresh = tf.ones_like(prob, tf.float32) * 0.5
        pred = tf.cast(tf.greater_equal(prob, thresh), tf.int32)
        true_pred = tf.cast(tf.equal(pred, labels), tf.float32)
        acc = tf.reduce_mean(true_pred)
        num_pos = tf.reduce_sum(tf.cast(labels, tf.float32))
        pos_inds = tf.where(tf.cast(labels, tf.bool))
        pos_pred = tf.reduce_sum(tf.gather(true_pred, pos_inds))
        rec = pos_pred / num_pos
        return acc, rec

    def metrics(self, ops={}):
        super(weightednet3, self).metrics(ops)
        ops['acc_weight'], ops['rec_weight'] = self.get_acc_rec(self.layers['rel_weight_prob'], self.data['rel_weight_labels'])
        ops['acc_fg'], ops['rec_fg'] = self.get_acc_rec(self.layers['fg_prob'],
                                                                self.data['fg_labels'])

    # Losses
    def losses(self):
        losses = super(weightednet3, self).losses()
        weight_loss = tf.losses.sigmoid_cross_entropy(tf.expand_dims(self.data['rel_weight_labels'], axis=1), self.layers['rel_weight_score'])
        fg_loss = tf.losses.sigmoid_cross_entropy(tf.expand_dims(self.data['fg_labels'], axis=1), self.layers['fg_score'])
        # y = tf.cast(self.data['fg_labels'], tf.float32)
        # p = tf.sigmoid(tf.squeeze(self.layers['fg_score']))
        # fg_loss =  - (1.0-y) * tf.log(1.0-p)
        # weight_loss = tf.reduce_mean(tf.square(self.layers['rel_weight_pred']-self.data['rel_weight_labels']))
        # weight_loss = tf.reduce_mean(tf.abs(self.layers['rel_weights'] - self.data['rel_weights']))
        # print(tf.expand_dims(self.data['fg_labels'], axis=1), self.layers['fg_score'])
        losses['loss_weight'] = tf.reduce_mean(weight_loss)
        losses['loss_fg'] = tf.reduce_mean(fg_loss)
        losses['loss_total'] = losses['loss_total'] + losses['loss_weight'] + losses['loss_fg']
        return losses