import tensorflow as tf
from networks.models import *
from fast_rcnn.config import cfg
import networks.net_utils as utils
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

class sptnet2(sptnet):
    def __init__(self, data):
        super(sptnet2, self).__init__(data)
        self.num_spatials = data['num_spatials']
        self.if_pred_spt = True
        self.if_pred_rel = True
        self.use_embedding = True
        self.use_spatial = True
        self.loss_weights['spt'] = 1

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], self.pooling_size, self.pooling_size,
                                             name='roi_conv_out')
        rel_roi_conv_out = self._net_roi_pooling([conv_net, self.rel_rois], self.pooling_size, self.pooling_size,
                                                 name='rel_roi_conv_out')
        roi_fc_out = self._net_roi_fc(roi_conv_out)
        self.rel_inx1, self.rel_inx2 = self._relation_indexes()

        if cfg.TRAIN.WEIGHT_REG:
            weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        else:
            weights_regularizer = tf.no_regularizer
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=weights_regularizer, ):

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
            vis = slim.dropout(net, keep_prob=self.keep_prob)
            vis = tf.concat([fc_sub, vis, fc_obj], axis=1)
            vis = slim.fully_connected(vis, size)
            vis = slim.dropout(vis, keep_prob=self.keep_prob)
            vis_feat = slim.fully_connected(vis, size)
            vis_feat = slim.dropout(vis_feat, keep_prob=self.keep_prob)
            if self.use_embedding:
                # class me
                sub_emb, obj_emb = self._class_feature(self.rel_inx1, self.rel_inx2)
                cls_emb = tf.concat([sub_emb, obj_emb], axis=1)
                cls_proj = slim.fully_connected(cls_emb, 128)
            if self.use_spatial:
                spt = self._spatial_feature(self.rel_inx1, self.rel_inx2)
            sub_feat = tf.concat([conv_sub, rel_roi_conv_out], axis=3)
            obj_feat = tf.concat([conv_obj, rel_roi_conv_out], axis=3)
            proj_sub = self._spatial_net(sub_feat)
            proj_obj = self._spatial_net(obj_feat, reuse=True)
            net_spt = proj_sub - proj_obj
            if self.if_pred_spt: self._spatial_pred(net_spt)

            rel_feat = tf.concat([vis_feat, cls_proj, spt, net_spt], axis=1)
            rel_feat = slim.fully_connected(rel_feat, size)
            rel_feat = slim.dropout(rel_feat, self.keep_prob)
            self._rel_pred(rel_feat)

    def _spatial_net(self, inputs, name='spt_feat', reuse=False):
        with tf.variable_scope(name, 'spt_feat', reuse=reuse):
            net = slim.conv2d(inputs, 256, [3, 3])
            net = slim.conv2d(net, 256, [3, 3])
            net = slim.conv2d(net, 256, [3, 3])
            net = slim.fully_connected(slim.flatten(net), 1024)
            net = slim.dropout(net, self.keep_prob)
            net = slim.fully_connected(net, 512)
            net = slim.dropout(net, self.keep_prob)
        return net

class sptnet3(sptnet):
    def __init__(self, data):
        super(sptnet3, self).__init__(data)
        self.num_spatials = data['num_spatials']
        self.if_pred_spt = False
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.loss_weights['spt'] = 1

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], 7, 7, name='roi_conv_out')
        roi_fc_out = self._net_roi_fc(roi_conv_out)
        rel_roi_conv_roi = self._net_roi_pooling([conv_net, self.rel_rois], 7, 7, name='rel_roi_conv_out')
        rel_roi_flatten = self._net_conv_reshape(rel_roi_conv_roi, name='rel_roi_flatten')
        rel_roi_fc_out = self._net_rel_roi_fc(rel_roi_flatten)
        # spatial info
        bbox = self.rois[:, 1:5]
        rel_inx1, rel_inx2 = self._relation_indexes()
        con_sub = tf.gather(roi_conv_out, rel_inx1)
        con_obj = tf.gather(roi_conv_out, rel_inx2)
        if self.if_pred_spt:
            sub_feat = tf.concat([con_sub, rel_roi_conv_roi], axis=3)
            obj_feat = tf.concat([con_obj, rel_roi_conv_roi], axis=3)
            proj_sub = self._spatial_feature(sub_feat)
            proj_obj = self._spatial_feature(obj_feat, reuse=True)
            spt = proj_sub - proj_obj
            self._spatial_pred(spt)
            # spt = super(sptnet2, self)._spatial_feature(rel_inx1, rel_inx2)
            # self._spatial_pred(spt)
        if self.if_pred_cls:
            self._cls_pred(roi_fc_out)
        if self.if_pred_rel:
            # rel visual feature
            con_sub = self._spatial_mask1(rel_inx1, rel_roi_conv_roi)
            con_obj = self._spatial_mask1(rel_inx2, rel_roi_conv_roi)
            shape = rel_roi_conv_roi.get_shape().as_list()
            con_sub.set_shape([shape[0], shape[1], shape[2], 1])
            con_obj.set_shape([shape[0], shape[1], shape[2], 1])
            print(con_sub.get_shape())
            net = tf.concat([con_sub, con_obj, rel_roi_conv_roi], axis=3)
            print(net.get_shape())
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            rel = slim.fully_connected(slim.flatten(net), 4096)
            rel = slim.dropout(rel, self.keep_prob)
            rel = slim.fully_connected(rel, 4096)
            rel = slim.dropout(rel, self.keep_prob)
            # class me
            cls_feat = self._class_feature(rel_inx1, rel_inx2)
            # if self.if_pred_spt:
            #     spt = tf.stop_gradient(spt, name='stop_spt_gradient')
            #     rel_feat = tf.concat([rel, cls_feat, spt], axis=1)
            # else: rel_feat = tf.concat([rel, cls_feat], axis=1)
            # rel_feat = slim.fully_connected(rel, 512)
            # rel_feat = slim.dropout(rel_feat, self.keep_prob)
            # rel_feat = slim.fully_connected(rel_feat, 512)
            # rel_feat = slim.dropout(rel_feat, self.keep_prob)
            self._rel_pred(rel)

    def _spatial_feature(self, inputs, name='spt_feat', reuse=False):
        with tf.variable_scope(name, 'spt_feat', reuse=reuse):
            net = slim.conv2d(inputs, 512, [3, 3])
            net = slim.conv2d(net, 128, [3, 3])
            net = slim.conv2d(net, 64, [3, 3])
            net = slim.fully_connected(slim.flatten(net), 512)
            net = slim.dropout(net, self.keep_prob)
            net = slim.fully_connected(net, 512)
            net = slim.dropout(net, self.keep_prob)
        return net
