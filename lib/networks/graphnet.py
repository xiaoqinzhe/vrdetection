import tensorflow as tf
from networks.models import *
from fast_rcnn.config import cfg
import networks.net_utils as utils
import tensorflow.contrib.slim as slim

class graphnet(basenet):
    def __init__(self, data):
        super(graphnet, self).__init__(data)
        self.obj_matrix = self.data['obj_matrix']
        self.rel_matrix = self.data['rel_matrix']
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox = False
        self.use_embedding = True
        self.rel_layer_suffix = {'_vis'}

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], self.pooling_size, self.pooling_size,
                                             name='roi_conv_out')
        rel_roi_conv_out = self._net_roi_pooling([conv_net, self.rel_rois], self.pooling_size, self.pooling_size,
                                                 name='rel_roi_conv_out')
        roi_flatten = self._net_conv_reshape(roi_conv_out, name='roi_flatten')
        roi_fc_out = self._net_roi_fc(roi_flatten)
        self.rel_inx1, self.rel_inx2 = self._relation_indexes()
        # spatial info
        bbox = self.rois[:, 1:5]
        if self.if_pred_rel:
            conv_sub = tf.gather(roi_conv_out, self.rel_inx1)
            conv_obj = tf.gather(roi_conv_out, self.rel_inx2)
            fc_sub = tf.gather(roi_fc_out, self.rel_inx1)
            fc_obj = tf.gather(roi_fc_out, self.rel_inx2)
            cls_sub, cls_obj = self._class_feature(self.rel_inx1, self.rel_inx2)
            cls_feat = slim.fully_connected(tf.concat([cls_sub, cls_obj], axis=1), 128)
            spt = self._spatial_feature(self.rel_inx1, self.rel_inx2)
            net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024)
            net = slim.dropout(net, keep_prob=self.keep_prob)
            net = slim.fully_connected(net, 1024)
            net = slim.dropout(net, keep_prob=self.keep_prob)

            # case 1
            # p_feature = tf.concat([fc_sub, net, fc_obj], axis=1)
            # p_feature = slim.fully_connected(p_feature, 4096)
            # p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            # p_feature = slim.fully_connected(p_feature, 4096)
            # p_feature = tf.concat([p_feature, cls_feat], axis=1)
            # self._rel_pred(p_feature, '_vis')
            # net = self.gcn(p_feature, self.rel_matrix, next_dim=2048, name="rel_gcn1", activation=tf.nn.leaky_relu)
            # net = self.gcn(net, self.rel_matrix, next_dim=2048, name="rel_gcn12", activation=tf.nn.leaky_relu)
            # net = self.gcn(net, self.rel_matrix, next_dim=1024, name="rel_gcn2", activation=tf.nn.leaky_relu)
            # net = self.gcn(net, self.rel_matrix, next_dim=1024, name="rel_gcn23", activation=tf.nn.leaky_relu)
            # net = self.gcn(net, self.rel_matrix, next_dim=512, name="rel_gcn3", activation=tf.nn.leaky_relu)
            # net = self.gcn(net, self.rel_matrix, next_dim=512, name="rel_gcn33", activation=tf.nn.leaky_relu)
            # net = tf.concat([p_feature, net], axis=1)
            # self._rel_pred(net)

            # case 2
            p_feature = tf.concat([fc_sub, net, fc_obj], axis=1)
            p_feature = slim.fully_connected(p_feature, 1024)
            p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            p_feature = slim.fully_connected(p_feature, 1024)
            p_feature = tf.concat([p_feature, cls_feat, spt], axis=1)
            self._rel_pred(p_feature, '_vis')
            self.rel_layer_suffix = {}
            net = self.gcn(p_feature, self.rel_matrix, next_dim=1024, name="rel_gcn1", activation=tf.nn.leaky_relu)
            net = self.gcn(net, self.rel_matrix, next_dim=1024, name="rel_gcn11", activation=tf.nn.leaky_relu)
            net = self.gcn(net, self.rel_matrix, next_dim=512, name="rel_gcn2", activation=tf.nn.leaky_relu)
            net = self.gcn(net, self.rel_matrix, next_dim=512, name="rel_gcn22", activation=tf.nn.leaky_relu)
            net = self.gcn(net, self.rel_matrix, next_dim=256, name="rel_gcn3", activation=tf.nn.leaky_relu)
            net = self.gcn(net, self.rel_matrix, next_dim=256, name="rel_gcn33", activation=tf.nn.leaky_relu)
            net = tf.concat([p_feature, net], axis=1)
            self._rel_pred(net)

            # case 3
            # p_feature = tf.concat([fc_sub, net, fc_obj], axis=1)
            # p_feature = slim.fully_connected(p_feature, 4096)
            # p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            # p_feature = slim.fully_connected(p_feature, 1024)
            # p_feature = tf.concat([p_feature, cls_feat], axis=1)
            # self._rel_pred(p_feature, '_vis')
            # net = self.gcn(p_feature, self.rel_matrix, next_dim=1024, name="rel_gcn1", activation=tf.nn.leaky_relu)
            # net = self.gcn(net, self.rel_matrix, next_dim=512, name="rel_gcn2", activation=tf.nn.leaky_relu)
            # net = self.gcn(net, self.rel_matrix, next_dim=256, name="rel_gcn3", activation=tf.nn.leaky_relu)
            # net = slim.fully_connected(net, 256)
            # net = tf.concat([p_feature, net], axis=1)
            # self._rel_pred(net)

            # case 3
            # cls_emb = self.gcn(self.cls_emb, self.obj_matrix, name="obj_gcn1")
            # cls_emb = self.gcn(cls_emb, self.obj_matrix, name="obj_gcn2")
            # cls_emb = self.gcn(cls_emb, self.obj_matrix, name="obj_gcn3")
            # cls_emb_sub = tf.gather(cls_emb, self.rel_inx1)
            # cls_emb_obj = tf.gather(cls_emb, self.rel_inx2)
            # cls_feat = slim.fully_connected(tf.concat([cls_emb_sub, cls_emb_obj], axis=1), 128)
            # p_feature = tf.concat([fc_sub, net, fc_obj], axis=1)
            # p_feature = slim.fully_connected(p_feature, 4096)
            # p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            # p_feature = slim.fully_connected(p_feature, 4096)
            # p_feature = tf.concat([p_feature, cls_feat], axis=1)
            # self._rel_pred(p_feature, '_vis')
            # net = self.gcn(p_feature, self.rel_matrix, next_dim=1024, name="rel_gcn1")
            # net = self.gcn(net, self.rel_matrix, next_dim=512, name="rel_gcn2")
            # net = self.gcn(net, self.rel_matrix, next_dim=256, name="rel_gcn3")
            # net = tf.concat([p_feature, net], axis=1)
            # self._rel_pred(net)

    def gcn(self, inputs, matrix, name, next_dim=None, reuse=False, activation=tf.sigmoid):
        with tf.variable_scope(name, reuse=reuse):
            if next_dim is None: next_dim = inputs.shape.as_list()[1]
            weight = tf.get_variable("weight", shape=[inputs.shape.as_list()[1], next_dim])
            net = tf.matmul(tf.matmul(matrix, inputs), weight)
            net = activation(net)
        return net

    def metrics(self, ops={}):
        if self.if_pred_cls:
            ops['acc_cls'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.layers['cls_prob'], axis=1, output_type=tf.int32), self.data['labels']), tf.float32))
        if self.if_pred_rel:
            ops['acc_rel'] = self.accuracy(self.layers['rel_pred'], self.predicates, "acc_rel", ignore_bg=cfg.TRAIN.USE_SAMPLE_GRAPH)
            for suffix in self.rel_layer_suffix:
                ops['acc_rel'+suffix] = self.accuracy(self.layers['rel_pred'+suffix], self.predicates, "acc_rel"+suffix,
                                               ignore_bg=cfg.TRAIN.USE_SAMPLE_GRAPH)

    # Losses
    def losses(self):
        losses = {}
        losses['loss_total'] = None
        if self.if_weight_reg:
            losses['loss_total'] = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
        if self.if_pred_rel:
            self._rel_losses(losses)
            if losses['loss_total'] == None:
                losses['loss_total'] = losses['loss_rel']
            else:
                losses['loss_total'] = tf.add(losses['loss_total'], losses['loss_rel'])
            for suffix in self.rel_layer_suffix:
                self._rel_losses(losses, suffix)
                losses['loss_total'] = tf.add(losses['loss_total'], losses['loss_rel'+suffix])
                losses['loss_total'] = losses['loss_total'] / (len(self.rel_layer_suffix)+1)
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
