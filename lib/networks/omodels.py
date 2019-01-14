import tensorflow as tf
from networks.models import *
from fast_rcnn.config import cfg
import networks.net_utils as utils
import tensorflow.contrib.slim as slim

class vtranse(basenet):
    def __init__(self, data):
        super(vtranse, self).__init__(data)
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox =False

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], 7, 7, name='roi_conv_out')
        roi_fc_out = self._net_roi_fc(roi_conv_out)
        # class me
        cls_pred = tf.one_hot(self.labels, depth=self.num_classes)
        cls_fc = slim.fully_connected(cls_pred, 256, scope="cls_emb")
        # spatial info
        bbox = self.rois[:, 1:5]
        if self.if_pred_rel:
            rel_inx1, rel_inx2 = self._relation_indexes()
            con_sub = tf.gather(roi_fc_out, rel_inx1)
            con_obj = tf.gather(roi_fc_out, rel_inx2)
            cls_sub = tf.gather(cls_fc, rel_inx1)
            cls_obj = tf.gather(cls_fc, rel_inx2)
            bbox_sub = tf.gather(bbox, rel_inx1)
            bbox_obj = tf.gather(bbox, rel_inx2)
            spatial_sub = self._relative_spatial(bbox_sub, bbox_obj)
            spatial_obj = self._relative_spatial(bbox_obj, bbox_sub)
            sub_feat = tf.concat([con_sub, cls_sub, spatial_sub], axis=1)
            obj_feat = tf.concat([con_obj, cls_obj, spatial_obj], axis=1)
            proj_sub = slim.fully_connected(sub_feat, 512)
            proj_obj = slim.fully_connected(obj_feat, 512)
            self._rel_pred(proj_sub - proj_obj)

class zoomnet(basenet):
    def __init__(self, data):
        super(zoomnet, self).__init__(data)
        self.num_spatials = data['num_spatials']
        self.if_pred_cls = False
        self.if_pred_rel = True
        self.if_pred_spt = False

    def _net(self):
        conv_sub, conv_obj, conv_rel = self._net_conv(self.ims)
        if self.if_pred_rel:
            # rel visual feature
            rel = slim.fully_connected(slim.flatten(conv_rel), 4096)
            rel = slim.dropout(rel, self.keep_prob)
            rel = slim.fully_connected(rel, 4096)
            rel = slim.dropout(rel, self.keep_prob)
            self._rel_pred(rel)

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
            if self.stop_gradient: net = tf.stop_gradient(net, name='stop_gradient')
            # pooling
            roi_conv_out = self._net_roi_pooling([net, self.rois], 24, 24, name='roi_conv_out')
            rel_roi_conv_out = self._net_roi_pooling([net, self.rel_rois], 24, 24, name='rel_roi_conv_out')
            rel_inx1, rel_inx2 = self._relation_indexes()
            conv_sub = tf.gather(roi_conv_out, rel_inx1)
            conv_obj = tf.gather(roi_conv_out, rel_inx2)
            conv_sub, conv_obj, conv_rel = self.sca_module(conv_sub, conv_obj, rel_roi_conv_out, self._conv4_12, self._conv4_3, rel_inx1, rel_inx2)
            conv_sub, conv_obj, conv_rel = self.sca_module(conv_sub, conv_obj, conv_rel, self._conv5_12,
                                                           self._conv5_3, rel_inx1, rel_inx2)
        return conv_sub, conv_obj, conv_rel

    def sca_module(self, conv_sub, conv_obj, rel_roi_conv_out, conv_func1, conv_func2, rel_inx1, rel_inx2):
        conv_sub = conv_func1(conv_sub, False)
        conv_obj = conv_func1(conv_obj, True)
        conv_rel = conv_func1(rel_roi_conv_out, True)
        spt_sub = self._spatial_conv(rel_inx1, conv_sub, conv_rel)
        spt_obj = self._spatial_conv(rel_inx2, conv_obj, conv_rel)
        shape = conv_rel.get_shape().as_list()
        spt_sub.set_shape([shape[0], shape[1], shape[2], shape[3]])
        spt_obj.set_shape([shape[0], shape[1], shape[2], shape[3]])
        rel1 = conv_func2((spt_sub + conv_rel) / 2.0, False)
        rel2 = conv_func2((spt_obj + conv_rel) / 2.0, True)
        rel3 = conv_func2((spt_sub + conv_obj) / 2.0, True)
        sub = conv_func2((spt_sub + conv_rel) / 2.0, True)
        obj = conv_func2((spt_obj + conv_rel) / 2.0, True)
        rel = (rel1 + rel2 + rel3) / 3.0
        return sub, obj, rel

    def _conv4_12(self, inputs, reuse=False):
        net = slim.conv2d(inputs, 512, [3, 3], scope='conv4/conv4_1', reuse=reuse)
        net = slim.conv2d(net, 512, [3, 3], scope='conv4/conv4_2', reuse=reuse)
        return net

    def _conv4_3(self, inputs, reuse=False):
        net = slim.conv2d(inputs, 512, [3, 3], scope='conv4/conv4_3', reuse=reuse)
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool4')
        return net

    def _conv5_12(self, inputs, reuse=False):
        net = slim.conv2d(inputs, 512, [3, 3], scope='conv5/conv5_1', reuse=reuse)
        net = slim.conv2d(net, 512, [3, 3], scope='conv5/conv5_2', reuse=reuse)
        return net

    def _conv5_3(self, inputs, reuse=False):
        net = slim.conv2d(inputs, 512, [3, 3], scope='conv5/conv5_3', reuse=reuse)
        return net

class zoomnet2(basenet):
    def __init__(self, data):
        super(zoomnet2, self).__init__(data)
        self.num_spatials = data['num_spatials']
        self.if_pred_cls = True
        self.if_pred_rel = True
        self.if_pred_spt = False

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
        rel_inx1, rel_inx2 = self._relation_indexes()
        con_sub = tf.gather(roi_conv_out, rel_inx1)
        con_obj = tf.gather(roi_conv_out, rel_inx2)
        if self.if_pred_cls:
            self._cls_pred(roi_fc_out)
        if self.if_pred_rel:
            # rel visual feature
            con_sub = self._spatial_conv(rel_inx1, con_sub, rel_roi_conv_roi)
            con_obj = self._spatial_conv(rel_inx2, con_obj, rel_roi_conv_roi)
            shape = rel_roi_conv_roi.get_shape().as_list()
            con_sub.set_shape([shape[0], shape[1], shape[2], shape[3]])
            con_obj.set_shape([shape[0], shape[1], shape[2], shape[3]])
            conv1 = slim.conv2d(tf.concat([con_sub, con_obj], axis=3), 512, [3, 3])
            conv2 = slim.conv2d(tf.concat([con_sub, rel_roi_conv_roi], axis=3), 512, [3, 3])
            conv3 = slim.conv2d(tf.concat([con_obj, rel_roi_conv_roi], axis=3), 512, [3, 3])
            net = tf.concat([conv1, conv2, conv3], axis=3)
            net = slim.conv2d(net, 1024, [3, 3])
            net = slim.conv2d(net, 1024, [3, 3])
            rel = slim.fully_connected(slim.flatten(net), 4096)
            rel = slim.dropout(rel, self.keep_prob)
            rel = slim.fully_connected(rel, 4096)
            rel = slim.dropout(rel, self.keep_prob)
            self._rel_pred(rel)

class visualnet(basenet):
    def __init__(self, data):
        super(visualnet, self).__init__(data)
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox =False

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], self.pooling_size, self.pooling_size, name='roi_conv_out')
        rel_roi_conv_out = self._net_roi_pooling([conv_net, self.rel_rois], self.pooling_size, self.pooling_size,
                                             name='rel_roi_conv_out')
        roi_fc_out = self._net_roi_fc(roi_conv_out)
        self.rel_inx1, self.rel_inx2 = self._relation_indexes()
        # class me
        cls_pred = tf.one_hot(self.labels, depth=self.num_classes)
        cls_fc = slim.fully_connected(cls_pred, 256, scope="cls_emb")
        # spatial info
        bbox = self.rois[:, 1:5]
        if self.if_pred_rel:
            fc_sub = tf.gather(roi_fc_out, self.rel_inx1)
            fc_obj = tf.gather(roi_fc_out, self.rel_inx2)
            # conv 1 2
            conv_sub = tf.gather(roi_conv_out, self.rel_inx1)
            conv_obj = tf.gather(roi_conv_out, self.rel_inx2)
            net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=self.keep_prob)
            net = slim.fully_connected(net, 4096)
            p_feature = slim.dropout(net, keep_prob=self.keep_prob)
            # 1
            # self._rel_pred(net)
            # 2
            vis_feat = tf.concat([fc_sub, p_feature,  fc_obj], axis=1)
            vis = slim.fully_connected(vis_feat, 4096)
            vis = slim.dropout(vis, keep_prob=self.keep_prob)
            vis = slim.fully_connected(vis, 4096)
            vis = slim.dropout(vis, keep_prob=self.keep_prob)
            self._rel_pred(vis)
            # 3, 4
            #
            # rel_roi_fc_out = self._net_rel_roi_fc(self._net_conv_reshape(rel_roi_conv_out))
            # # rel_feat = rel_roi_fc_out # 3
            # rel_feat = tf.concat([fc_sub, rel_roi_fc_out, fc_obj], axis=1) # 4
            # self._rel_pred(rel_feat)

class multinet(basenet):
    def __init__(self, data):
        super(multinet, self).__init__(data)
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox =False
        self.use_embedding = True
        self.use_spatial = True
        self.embedded_size = 128

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], self.pooling_size, self.pooling_size, name='roi_conv_out')
        rel_roi_conv_out = self._net_roi_pooling([conv_net, self.rel_rois], self.pooling_size, self.pooling_size,
                                             name='rel_roi_conv_out')
        roi_fc_out = self._net_roi_fc(roi_conv_out)
        self.rel_inx1, self.rel_inx2 = self._relation_indexes()
        # spatial info
        bbox = self.rois[:, 1:5]
        if self.if_pred_rel:
            if cfg.TRAIN.WEIGHT_REG:
                weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
            else: weights_regularizer = tf.no_regularizer
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,):

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

                # case 1   44
                self._rel_pred(cls_proj)

                # case 2   38.8
                # self._rel_pred(spt)

                # case 3   49
                # self._rel_pred(vis_feat)

                # case 4    52.6
                # net = tf.concat([vis_feat, cls_proj], axis=1)
                # self._rel_pred(net)

                # case 5   51.2
                # net = tf.concat([vis_feat, spt], axis=1)
                # self._rel_pred(net)

                # case 6
                net = tf.concat([vis_feat, cls_proj, spt], axis=1)
                self._rel_pred(net)

                # case 7
                # net = tf.concat([vis_feat, cls_proj], axis=1)
                # with tf.variable_scope('rel_score'):
                #     weight = tf.get_variable("weight", shape=[net.shape.as_list()[1], self.num_predicates])
                # rel_score = tf.matmul(net, weight)
                # self.layers['rel_score'] = rel_score
                # self.layers['rel_prob'] = slim.softmax(rel_score, scope='rel_prob')
                # self.layers['rel_pred'] = tf.argmax(self.layers['rel_prob'], axis=1, name='rel_pred')

                # case 8  48.7
                # rel_roi_fc_out = self._net_rel_roi_fc(self._net_conv_reshape(rel_roi_conv_out))
                # rel_feat = tf.concat([fc_sub, rel_roi_fc_out, fc_obj], axis=1) # 4
                # vis = slim.fully_connected(rel_feat, 4096)
                # vis = slim.dropout(vis, keep_prob=self.keep_prob)
                # vis = slim.fully_connected(vis, 4096)
                # vis_feat = slim.dropout(vis, keep_prob=self.keep_prob)
                # self._rel_pred(tf.concat([vis_feat, cls_proj], axis=1))

                # case 9
                # rel_roi_fc_out = self._net_rel_roi_fc(self._net_conv_reshape(rel_roi_conv_out))
                # rel_feat = tf.concat([fc_sub, rel_roi_fc_out, fc_obj], axis=1)  # 4
                # vis = slim.fully_connected(rel_feat, 4096)
                # vis = slim.dropout(vis, keep_prob=self.keep_prob)
                # vis = slim.fully_connected(vis, 4096)
                # vis_feat = slim.dropout(vis, keep_prob=self.keep_prob)
                # self._rel_pred(tf.concat([vis_feat, cls_proj, spt], axis=1))
