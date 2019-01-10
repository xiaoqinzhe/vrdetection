import tensorflow as tf
from models import *
from fast_rcnn.config import cfg
import net_utils as utils
import tensorflow.contrib.slim as slim

class cainet(basenet):
    def __init__(self, data):
        super(cainet, self).__init__(data)
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox = False
        self.use_embedding = True
        self.use_spatial = True
        self.embedded_size = 64

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
            print(self.num_rel)
            cls_emb = self.cls_emb()
            vis = self.visual_feat(rel_roi_conv_out, use_ctx_att=True, cls_emb=cls_emb)
            print(vis.shape)
            spt = self.spt_feat()
            print(spt.shape)
            feat = tf.concat([vis, spt], axis=2)
            self.rel_pred(feat, cls_emb)

    def visual_feat(self, conv_feat, use_ctx_att=True, cls_emb=None):
        with tf.variable_scope('attention'):
            conv_shape = conv_feat.get_shape()
            conv_feat = tf.reshape(conv_feat, [-1, conv_shape[1]*conv_shape[2], conv_shape[3]])
            if use_ctx_att:
                w_a = tf.get_variable("att_w_pa", [self.num_predicates, conv_shape[3]])
                v_a = tf.get_variable("att_v_pa", [cls_emb.get_shape()[1], self.num_predicates, conv_shape[3]])
                w_a = tf.tile(tf.expand_dims(w_a, axis=0), [tf.shape(conv_feat)[0], 1, 1])

                w = w_a + tf.reshape(tf.matmul(cls_emb, tf.reshape(v_a, [cls_emb.get_shape()[1], self.num_predicates*conv_shape[3]])),
                                     [-1, self.num_predicates, conv_shape[3]])  #(?, P, C)
                w = tf.tile(tf.expand_dims(w, axis=2), [1, 1, conv_shape[1]*conv_shape[2], 1])
                h = tf.tile(tf.expand_dims(conv_feat, axis=1), [1, self.num_predicates, 1, 1])
                bias = tf.get_variable("bias", shape=[1])
                att_vis_feat = tf.reduce_mean(w * h + bias, axis=2)
            else:
                w = tf.get_variable("weight", shape=[conv_shape[3], 1])
                context = conv_feat
                bias = tf.get_variable("bias", shape=[1])
                weights = tf.matmul(context, w) + bias
                weights = tf.nn.softmax(tf.squeeze(weights))
                weights = tf.reshape(tf.tile(weights, [1, conv_shape[3]]), [-1, conv_shape[1]*conv_shape[2], conv_shape[3]])
                att_vis_feat = tf.reduce_mean(weights * conv_feat, axis=1)
                att_vis_feat = tf.tile(tf.expand_dims(att_vis_feat, axis=1), [1, self.num_predicates, 1])
        return att_vis_feat    # (?, P, C)

    def spt_feat(self):
        spt = self._spatial_feature(self.rel_inx1, self.rel_inx2)
        spt = tf.tile(tf.expand_dims(spt, axis=1), [1, self.num_predicates, 1])
        return spt

    def cls_emb(self):
        sub_emb, obj_emb = self._class_feature(self.rel_inx1, self.rel_inx2)
        cls_emb = tf.concat([sub_emb, obj_emb], axis=1)
        cls_proj = slim.fully_connected(cls_emb, self.embedded_size, activation_fn=tf.nn.relu)
        return cls_proj

    def rel_pred(self, feat, cls_emb):
        feat_size = feat.shape.as_list()[2]
        with tf.variable_scope('rel_pred'):
            Vp = tf.get_variable("vp", shape=[self.embedded_size, self.num_predicates, feat_size])
            rp = tf.reshape(tf.matmul(cls_emb, tf.reshape(Vp,[self.embedded_size, self.num_predicates*feat_size])),
                            [-1, self.num_predicates, feat_size])
            w_p = tf.get_variable("w_p", shape=[self.num_predicates, feat_size])
            w_p = tf.tile(tf.expand_dims(w_p, axis=0), [tf.shape(cls_emb)[0], 1, 1])
            wp = w_p + rp
            rel = tf.reduce_sum(wp*feat, axis=2)
            self.layers['rel_score'] = rel
            self.layers['rel_prob'] = slim.softmax(rel, scope='rel_prob')
            self.layers['rel_pred'] = tf.argmax(self.layers['rel_prob'], axis=1, name='rel_pred')
