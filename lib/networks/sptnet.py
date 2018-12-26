import tensorflow as tf
from models import *
from fast_rcnn.config import cfg
import net_utils as utils
import tensorflow.contrib.slim as slim

class sptnet(basenet):
    def __init__(self, data):
        super(sptnet, self).__init__(data)
        self.num_spatials = data['num_spatials']
        self.if_pred_spt = True
        self.if_pred_rel = True
        self.loss_weights['spt'] = 1.0

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], self.pooling_size, self.pooling_size, name='roi_conv_out')
        rel_roi_conv_out = self._net_roi_pooling([conv_net, self.rel_rois], self.pooling_size, self.pooling_size,
                                             name='rel_roi_conv_out')
        roi_flatten = self._net_conv_reshape(roi_conv_out, name='roi_flatten')
        roi_fc_out = self._net_roi_fc(roi_flatten)
        self.rel_inx1, self.rel_inx2 = self._relation_indexes()
        # spatial info
        bbox = self.rois[:, 1:5]
        if self.if_pred_spt:
            conv_sub = tf.gather(roi_conv_out, self.rel_inx1)
            conv_obj = tf.gather(roi_conv_out, self.rel_inx2)
            fc_sub = tf.gather(roi_fc_out, self.rel_inx1)
            fc_obj = tf.gather(roi_fc_out, self.rel_inx2)
            cls_sub, cls_obj = self._class_feature(self.rel_inx1, self.rel_inx2)
            cls_feat = self.cls_feature_layer(cls_sub, cls_obj)
            spt = self._spatial_feature(self.rel_inx1, self.rel_inx2)

            # case 1: direction
            # net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
            # net = self.p_feature_layer(net)
            # p_feature = self.ctx_feature_layer(fc_sub, net, fc_obj)
            # self._spatial_pred(p_feature)

            # case 2: direction + spt
            # net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
            # net = self.p_feature_layer(net)
            # p_feature = self.ctx_feature_layer(fc_sub, net, fc_obj)
            # p_feature = tf.concat([p_feature, spt], axis=1)
            # self._spatial_pred(p_feature)

            # case 3: direction + spt
            # net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
            # spt_feat = self.p_feature_layer(net)
            # spt_feat = self.ctx_feature_layer(fc_sub, spt_feat, fc_obj)
            # spt_feat = tf.concat([spt_feat, spt], axis=1)
            # self._spatial_pred(spt_feat)
            # # rel feat
            # rel_feat = self.p_feature_layer(net, name="rel_feat1")
            # rel_feat = self.ctx_feature_layer(fc_sub, rel_feat, fc_obj, name="rel_feat2")
            # spt_feat = self.spt_feature(inputs=self.layers['spt_pred'], gt=False)
            # feat = tf.concat([rel_feat, spt_feat, cls_feat], axis=1)
            # self._rel_pred(feat)

            # case 4: direction + spt
            net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
            spt_feat = self.p_feature_layer(net)
            spt_feat = self.ctx_feature_layer(fc_sub, spt_feat, fc_obj)
            spt_feat = tf.concat([spt_feat, spt], axis=1)
            self._spatial_pred(spt_feat)
            # rel feat
            rel_feat = self.p_feature_layer(net, name="rel_feat1")
            rel_feat = self.ctx_feature_layer(fc_sub, rel_feat, fc_obj, name="rel_feat2")
            spt_feat = slim.fully_connected(self.layers['spt_prob'], 16, activation_fn=None)
            feat = tf.concat([rel_feat, spt_feat, cls_feat], axis=1)
            self._rel_pred(feat)

            # case 4: using gt spt 70%
            # self.if_pred_spt=False
            # net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
            # spt_feat = self.gt_spt_feature()
            # # rel feat
            # rel_feat = self.p_feature_layer(net, name="rel_feat1")
            # rel_feat = self.ctx_feature_layer(fc_sub, rel_feat, fc_obj, name="rel_feat2")
            # feat = tf.concat([rel_feat, spt_feat, cls_feat], axis=1)
            # self._rel_pred(feat)


            # case 3: direction + transE, acc 44
            # net = tf.concat([conv_sub, rel_roi_conv_out], axis=3)
            # net = self.p_feature_layer(net)
            # sub_pos = self.project_layer(net, proj_size=512)
            # net = tf.concat([conv_obj, rel_roi_conv_out], axis=3)
            # net = self.p_feature_layer(net, reuse=True)
            # obj_pos = self.project_layer(net, proj_size=512, reuse=True)
            # self._spatial_pred(sub_pos-obj_pos)

            # p_feature = tf.concat([p_feature, cls_feat], axis=1)
            # self._rel_pred(p_feature, '_vis')
            # # reversed p feature
            # net = tf.concat([conv_obj, rel_roi_conv_out, conv_sub], axis=3)
            # net = self.p_feature_layer(net, reuse=True)
            # re_p_feature = self.ctx_feature_layer(fc_obj, net, fc_sub, reuse=True)
            # re_cls_feat = self.cls_feature_layer(cls_obj, cls_sub, reuse=True)
            # re_p_feature = tf.concat([re_p_feature, re_cls_feat], axis=1)
            # net = tf.concat([p_feature, re_p_feature], axis=1)
            # self._rel_pred(net)

    def spt_feature(self, inputs=None, gt=True):
        if gt: inputs = self.data['rel_spts']
        oh = tf.one_hot(inputs, depth=8)
        spt_feat = slim.fully_connected(oh, 16, activation_fn=None)
        return spt_feat


    def p_feature_layer(self, p_conv, name="p_feature_layer", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            net = slim.conv2d(p_conv, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=self.keep_prob)
            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=self.keep_prob)
        return net

    def project_layer(self, input, proj_size=3, reuse=False):
        with tf.variable_scope("project_layer", reuse=reuse):
            pos = slim.fully_connected(input, 512)
            #p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            pos = slim.fully_connected(pos, 512)
            #p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            pos = slim.fully_connected(pos, proj_size, activation_fn=None)
        return pos

    def ctx_feature_layer(self, fc_sub, p, fc_obj, name="ctx_feature_layer", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            p_feature = tf.concat([fc_sub, p, fc_obj], axis=1)
            p_feature = slim.fully_connected(p_feature, 4096)
            p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            p_feature = slim.fully_connected(p_feature, 4096)
            p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
        return p_feature

    def cls_feature_layer(self, cls_sub, cls_obj, reuse=False):
        with tf.variable_scope("cls_feature_layer", reuse=reuse):
            cls_feat = slim.fully_connected(tf.concat([cls_sub, cls_obj], axis=1), 256)
        return cls_feat

    def _spatial_pred(self, inputs):
        net = slim.fully_connected(inputs, self.num_spatials, activation_fn=None, scope='spt_score')
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
