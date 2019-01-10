import tensorflow as tf
from networks.models import *
from fast_rcnn.config import cfg
import networks.net_utils as utils
import tensorflow.contrib.slim as slim

class attnet(basenet):
    def __init__(self, data):
        super(attnet, self).__init__(data)
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox =False

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], self.pooling_size, self.pooling_size, name='roi_conv_out')
        rel_roi_conv_out = self._net_roi_pooling([conv_net, self.rel_rois], self.pooling_size, self.pooling_size,
                                             name='rel_roi_conv_out')
        roi_flatten = self._net_conv_reshape(roi_conv_out, name='roi_flatten')
        roi_fc_out = self._net_roi_fc(roi_flatten)
        rel_inx1, rel_inx2 = self._relation_indexes()
        # class me
        cls_pred = tf.one_hot(self.labels, depth=self.num_classes)
        cls_fc = slim.fully_connected(cls_pred, 256, scope="cls_emb")
        # spatial info
        bbox = self.rois[:, 1:5]
        if self.if_pred_rel:
            conv_sub = tf.gather(roi_flatten, rel_inx1)
            conv_obj = tf.gather(roi_flatten, rel_inx2)
            context = tf.concat([conv_sub, conv_obj], axis=1)
            # net = self.local_attention(context, rel_roi_conv_out)
            # use context
            net = self._net_rel_roi_fc(self._net_conv_reshape(rel_roi_conv_out))
            net = tf.concat([conv_sub, conv_obj, net], axis=1)
            # net = slim.conv2d(net, 512, [3, 3])
            # net = slim.conv2d(net, 512, [3, 3])
            # net = slim.conv2d(net, 512, [3, 3])
            # net = slim.flatten(net)
            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=self.keep_prob)
            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=self.keep_prob)
            self._rel_pred(net)

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

class ctxnet(attnet):
    def __init__(self, data):
        super(ctxnet, self).__init__(data)
        self.obj_context_o = data['obj_context_o']
        self.obj_context_p = data['obj_context_p']
        self.obj_context_inds = data['obj_context_inds']
        self.rel_context = data['rel_context']
        self.rel_context_inds = data['rel_context_inds']
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox =False
        self.use_embedding = True
        # self.rel_layer_suffix = {'_vis'}
        self.rel_layer_suffix = {}

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
            conv_sub = tf.gather(roi_conv_out, self.rel_inx1)
            conv_obj = tf.gather(roi_conv_out, self.rel_inx2)
            roi_fc_emb = slim.fully_connected(roi_fc_out, 1024)
            fc_sub = tf.gather(roi_fc_emb, self.rel_inx1)
            fc_obj = tf.gather(roi_fc_emb, self.rel_inx2)
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
            # p_feature = slim.fully_connected(p_feature, 1024)
            # p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            # p_feature = slim.fully_connected(p_feature, 1024)
            # p_value = tf.concat([p_feature, cls_feat, spt], axis=1)
            # p_value = slim.fully_connected(p_value, 1024)
            # p_key = p_value
            # # self.rel_layer_suffix = {'_vis'}
            # # self._rel_pred(p_value, '_vis')
            # o_feature = roi_fc_out
            # ctx_pred, ctx_obj = self.global_context_propagation(o_feature, p_value, p_key)
            #
            # net = p_value + ctx_pred
            # net = tf.concat([net, cls_feat, spt], axis=1)
            # self._rel_pred(net)

            # case 2
            p_feature = tf.concat([fc_sub, net, fc_obj], axis=1)
            p_feature = slim.fully_connected(p_feature, 1024)
            p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            p_feature = slim.fully_connected(p_feature, 1024)
            p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            # p_value = tf.concat([p_feature, cls_feat, spt], axis=1)
            # p_value = slim.fully_connected(p_value, 1024)
            p_value=p_feature
            p_key = p_feature
            # self.rel_layer_suffix = {'_vis'}
            # self._rel_pred(p_value, '_vis')
            o_feature = roi_fc_out
            ctx_pred, ctx_obj = self.global_context_propagation(o_feature, p_value, p_key, cls_feat=cls_feat, spt=spt)

            net = tf.concat([p_value, ctx_pred], axis=1)
            # net=p_value+ctx_pred
            # net = slim.fully_connected(net, 1024)
            # net = slim.fully_connected(net, 1024)
            net = tf.concat([net, cls_feat, spt], axis=1)
            self._rel_pred(net)


            # self.rel_layer_suffix = {'_vis'}
            # spt_feat = self._spatial_feature(self.rel_inx1, self.rel_inx2)
            # p_feature = tf.concat([fc_sub, net, fc_obj], axis=1)
            # p_feature = slim.fully_connected(p_feature, 4096)
            # p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            # p_feature = slim.fully_connected(p_feature, 1024)
            # p_value = tf.concat([p_feature, cls_feat], axis=1)
            # p_key = p_value
            # o_feature = roi_fc_out
            # self._rel_pred(p_value, '_vis')
            # ctx_pred, ctx_obj = self.global_context_propagation(o_feature, p_value, p_key, enc=True)
            # p_value = p_value + ctx_pred
            # ctx_pred, ctx_obj = self.global_context_propagation(o_feature, p_value, p_value, enc=True)
            # p_value = p_value + ctx_pred
            # ctx_pred, ctx_obj = self.global_context_propagation(o_feature, p_value, p_value, enc=True)
            # p_value = p_value + ctx_pred
            # self._rel_pred(p_value)

            # case 3
            # p_feature = tf.concat([fc_sub, net, fc_obj, cls_feat], axis=1)
            # p_feature = slim.fully_connected(p_feature, 4096)
            # p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            # p_feature = slim.fully_connected(p_feature, 4096)
            # self._rel_pred(p_feature)
            # net = tf.concat([p_value, ctx_pred], axis=1)
            # net = slim.fully_connected(net, 4096)
            # net = slim.dropout(net, keep_prob=self.keep_prob)
            # self._rel_pred(net)

    def global_context_propagation(self, o_feature, p_value, p_key, cls_feat=None, spt=None, enc=False):
        # relation propagation

        # if enc:
        #     enc_p_feature = slim.fully_connected(p_feature, p_feature.shape.as_list()[1])
        #     pred_value = self.pad_and_gather(enc_p_feature, self.rel_context)
        # else: pred_value = self.pad_and_gather(p_feature, self.rel_context)
        # pred_ctx = self.pad_and_gather(p_key, self.rel_context)
        # pred_gather = tf.gather(p_key, self.rel_context_inds)
        # key = tf.concat([pred_gather, pred_ctx], axis=1)
        # pred_ctxv = self.pad_and_gather(p_feature, self.rel_context)
        # pred_gatherv = tf.gather(p_feature, self.rel_context_inds)
        # value = tf.concat([pred_gatherv, pred_ctxv], axis=1)
        # # attention
        # weights1 = slim.fully_connected(key, 1, activation_fn=tf.nn.relu)
        # weights2 = slim.fully_connected(value, 1, activation_fn=tf.nn.tanh)
        # weighted_preds = tf.multiply(weights1*weights2, pred_value)
        # ctx_pred = tf.segment_max(weighted_preds, self.rel_context_inds)

        if enc:
            enc_p_feature = slim.fully_connected(p_value, p_value.shape.as_list()[1])
            pred_value = self.pad_and_gather(enc_p_feature, self.rel_context)
        else: pred_value = self.pad_and_gather(p_value, self.rel_context)
        pred_ctx = self.pad_and_gather(p_key, self.rel_context)
        pred_gather = tf.gather(p_key, self.rel_context_inds)
        ctxs = tf.concat([pred_gather, pred_ctx], axis=1)
        # attention
        net = ctxs
        net = slim.fully_connected(ctxs, 1024)
        # net = slim.fully_connected(net, 512)
        weights = slim.fully_connected(net, 1)
        if(cls_feat!=None):
            cls_ctx = self.pad_and_gather(cls_feat, self.rel_context)
            cls_gather = tf.gather(cls_feat, self.rel_context_inds)
            cls_ctxs = tf.concat([cls_gather, cls_ctx], axis=1)
            cls_weights = slim.fully_connected(cls_ctxs, 128)
            cls_weights = slim.fully_connected(cls_weights, 1)
            weights = tf.multiply(weights, cls_weights)
        if (spt != None):
            spt_ctx = self.pad_and_gather(spt, self.rel_context)
            spt_gather = tf.gather(spt, self.rel_context_inds)
            spt_ctxs = tf.concat([spt_gather, spt_ctx], axis=1)
            spt_weights = slim.fully_connected(spt_ctxs, 1)
            weights = tf.multiply(weights, spt_weights)
        weighted_preds = tf.multiply(weights, pred_value)
        print(weights.shape, pred_value.shape)
        ctx_pred = tf.segment_sum(weighted_preds, self.rel_context_inds)

        # object propagation
        obj_gather = tf.gather(o_feature, self.obj_context_inds)
        obj = self.pad_and_gather(o_feature, self.obj_context_o)
        pred = self.pad_and_gather(p_value, self.obj_context_p)
        obj_ctx = tf.concat([obj, pred], axis=1)
        ctxs = tf.concat([obj_gather, obj_ctx], axis=1)
        # attention
        net = ctxs
        # net = slim.fully_connected(ctxs, 512)
        # net = slim.fully_connected(net, 512)
        weights = slim.fully_connected(net, 1)
        weighted_objs = tf.multiply(weights, obj_ctx)
        ctx_obj = tf.segment_sum(weighted_objs, self.obj_context_inds)

        return ctx_pred, ctx_obj

    def local_attention(self, context, conv_feat, name='local_attention', reuse = False):
        with tf.variable_scope(name, 'local_attention', reuse=reuse):
            conv_shape = conv_feat.get_shape()
            conv_feat = tf.reshape(conv_feat, [-1, conv_shape[1]*conv_shape[2], conv_shape[3]])
            cont_resp = tf.reshape(tf.tile(context, [1, conv_shape[1]*conv_shape[2]]), shape=[-1, conv_shape[1]*conv_shape[2], context.get_shape()[1]])
            input = tf.concat([conv_feat, cont_resp], axis=2)
            net = slim.fully_connected(input, 2048, scope='att1')
            net = slim.dropout(net, keep_prob=self.keep_prob)
            net = slim.fully_connected(net, 2048, scope='att2')
            net = slim.dropout(net, keep_prob=self.keep_prob)
            net = slim.fully_connected(net, 1, scope='att3')
            weights = tf.squeeze(net, axis=2)
            weights = tf.nn.softmax(weights)
            weights = tf.reshape(tf.tile(weights, [1, conv_feat.get_shape()[2]]),
                                 [-1, conv_feat.get_shape()[1], conv_feat.get_shape()[2]])
            net = tf.reduce_mean(weights * conv_feat, axis=1)
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

    def pad_and_gather(self, vecs, mask_inds, pad=None):
        """
        pad a vector with a zero row and gather with input inds
        """
        if pad is None:
            pad = tf.expand_dims(tf.zeros_like(vecs[0]), 0)
        else:
            pad = tf.expand_dims(pad, 0)
        vecs_padded = tf.concat([vecs, pad], 0)
        # flatten mask and edges
        vecs_gathered = tf.gather(vecs_padded, mask_inds)
        return vecs_gathered

    def padded_segment_reduce(self, vecs, segment_inds, num_segments, reduction_mode):
        """
        Reduce the vecs with segment_inds and reduction_mode
        Input:
            vecs: A Tensor of shape (batch_size, vec_dim)
            segment_inds: A Tensor containing the segment index of each
            vec row, should agree with vecs in shape[0]
        Output:
            A tensor of shape (vec_dim)
        """
        if reduction_mode == 'max':
            print('USING MAX POOLING FOR REDUCTION!')
            vecs_reduced = tf.segment_max(vecs, segment_inds)
        elif reduction_mode == 'mean':
            print('USING AVG POOLING FOR REDUCTION!')
            vecs_reduced = tf.segment_mean(vecs, segment_inds)
        vecs_reduced.set_shape([num_segments, vecs.get_shape()[1]])
        return vecs_reduced


class ctxnet2(ctxnet):
    def __init__(self, data):
        super(ctxnet2, self).__init__(data)
        self.obj_context_o = data['obj_context_o']
        self.obj_context_p = data['obj_context_p']
        self.obj_context_inds = data['obj_context_inds']
        self.rel_context = data['rel_context']
        self.rel_context_inds = data['rel_context_inds']
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox =False
        self.rel_layer_suffix = {}

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
        if self.if_pred_rel:
            conv_sub = tf.gather(roi_conv_out, self.rel_inx1)
            conv_obj = tf.gather(roi_conv_out, self.rel_inx2)
            fc_sub = tf.gather(roi_fc_out, self.rel_inx1)
            fc_obj = tf.gather(roi_fc_out, self.rel_inx2)
            cls_sub, cls_obj = self._class_feature(self.rel_inx1, self.rel_inx2)
            net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
            net = self.p_feature_layer(net)
            p_feature = self.ctx_feature_layer(fc_sub, net, fc_obj)
            cls_feat = self.cls_feature_layer(cls_sub, cls_obj)
            p_feature = tf.concat([p_feature, cls_feat], axis=1)
            self._rel_pred(p_feature, '_vis')
            # reversed p feature
            net = tf.concat([conv_obj, rel_roi_conv_out, conv_sub], axis=3)
            net = self.p_feature_layer(net, reuse=True)
            re_p_feature = self.ctx_feature_layer(fc_obj, net, fc_sub, reuse=True)
            re_cls_feat = self.cls_feature_layer(cls_obj, cls_sub, reuse=True)
            re_p_feature = tf.concat([re_p_feature, re_cls_feat], axis=1)
            net = tf.concat([p_feature, re_p_feature], axis=1)
            self._rel_pred(net)

    def p_feature_layer(self, p_conv, reuse=False):
        with tf.variable_scope("p_feature_layer", reuse=reuse):
            net = slim.conv2d(p_conv, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=self.keep_prob)
            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=self.keep_prob)
        return net

    def ctx_feature_layer(self, fc_sub, p, fc_obj, reuse=False):
        with tf.variable_scope("ctx_feature_layer", reuse=reuse):
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

class grunet(ctxnet):
    def __init__(self, data):
        super(grunet, self).__init__(data)
        self.obj_context_o = data['obj_context_o']
        self.obj_context_p = data['obj_context_p']
        self.obj_context_inds = data['obj_context_inds']
        self.rel_context = data['rel_context']
        self.rel_context_inds = data['rel_context_inds']
        self.if_pred_rel = True
        self.if_pred_cls = False
        self.if_pred_bbox =False
        self.rel_layer_suffix = {'_vis'}
        # intiialize lstms
        self.vert_state_dim = 512
        self.edge_state_dim = 512
        self.vert_rnn = tf.nn.rnn_cell.GRUCell(self.vert_state_dim, activation=tf.tanh)
        self.edge_rnn = tf.nn.rnn_cell.GRUCell(self.edge_state_dim, activation=tf.tanh)
        # lstm states
        self.vert_state = self.vert_rnn.zero_state(self.num_roi, tf.float32)
        self.edge_state = self.edge_rnn.zero_state(self.num_rel, tf.float32)

    def _net(self):
        conv_net = self._net_conv(self.ims)
        self.layers['conv_out'] = conv_net
        roi_conv_out = self._net_roi_pooling([conv_net, self.rois], self.pooling_size, self.pooling_size, name='roi_conv_out')
        rel_roi_conv_out = self._net_roi_pooling([conv_net, self.rel_rois], self.pooling_size, self.pooling_size,
                                             name='rel_roi_conv_out')
        roi_flatten = self._net_conv_reshape(roi_conv_out, name='roi_flatten')
        roi_fc_out = self._net_roi_fc(roi_flatten)
        self.rel_inx1, self.rel_inx2 = self._relation_indexes()
        # class me
        cls_sub, cls_obj = self._class_feature(self.rel_inx1, self.rel_inx2)
        cls_feat = slim.fully_connected(tf.concat([cls_sub, cls_obj], axis=1), 128)
        # spatial info
        bbox = self.rois[:, 1:5]
        if self.if_pred_rel:
            conv_sub = tf.gather(roi_conv_out, self.rel_inx1)
            conv_obj = tf.gather(roi_conv_out, self.rel_inx2)
            fc_sub = tf.gather(roi_fc_out, self.rel_inx1)
            fc_obj = tf.gather(roi_fc_out, self.rel_inx2)
            # context = tf.concat([conv_sub, conv_obj], axis=3)
            # net = self.local_attention(context, rel_roi_conv_out)
            # use context
            net = tf.concat([conv_sub, rel_roi_conv_out, conv_obj], axis=3)
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.conv2d(net, 512, [3, 3])
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096)
            net = slim.dropout(net, keep_prob=self.keep_prob)
            net = slim.fully_connected(net, 4096)
            p_feature = tf.concat([fc_sub, net, fc_obj], axis=1)
            p_feature = slim.fully_connected(p_feature, 4096)
            p_feature = slim.dropout(p_feature, keep_prob=self.keep_prob)
            p_feature = slim.fully_connected(p_feature, 1024)
            p_feature = tf.concat([p_feature, cls_feat], axis=1)
            o_feature = roi_fc_out
            self.vert_state = slim.fully_connected(o_feature, self.vert_state_dim)
            self.edge_state = slim.fully_connected(p_feature, self.edge_state_dim)
            self._rel_pred(self.edge_state, '_vis')
            ctx_pred, ctx_obj = self.global_context_propagation(self.vert_state, self.edge_state)
            ctx_obj = slim.fully_connected(ctx_obj, self.vert_state_dim)
            ctx_pred = slim.fully_connected(ctx_pred, self.vert_state_dim)
            self._vert_rnn_forward(ctx_obj, reuse=False)
            self._edge_rnn_forward(ctx_pred, reuse=False)
            ctx_pred, ctx_obj = self.global_context_propagation(self.vert_state, self.edge_state)
            ctx_obj = slim.fully_connected(ctx_obj, self.vert_state_dim)
            ctx_pred = slim.fully_connected(ctx_pred, self.vert_state_dim)
            ctx_obj = self._vert_rnn_forward(ctx_obj, reuse=True)
            ctx_pred = self._edge_rnn_forward(ctx_pred, reuse=True)
            self._rel_pred(self.edge_state)
            # self._cls_pred(ctx_obj)

    def global_context_propagation(self, o_feature, p_feature, p_key, enc=False):
        # relation propagation

        # encode p_feature
        if enc:
            enc_p_feature = slim.fully_connected(p_feature, p_feature.shape.as_list()[1])
            pred_value = self.pad_and_gather(enc_p_feature, self.rel_context)
        else: pred_value = self.pad_and_gather(p_feature, self.rel_context)
        pred_ctx = self.pad_and_gather(p_key, self.rel_context)
        pred_gather = tf.gather(p_key, self.rel_context_inds)
        key = tf.concat([pred_gather, pred_ctx], axis=1)
        pred_ctxv = self.pad_and_gather(p_feature, self.rel_context)
        pred_gatherv = tf.gather(p_feature, self.rel_context_inds)
        value = tf.concat([pred_gatherv, pred_ctxv], axis=1)
        # attention
        weights1 = slim.fully_connected(key, 1, activation_fn=tf.nn.relu)
        weights2 = slim.fully_connected(value, 1, activation_fn=tf.nn.tanh)
        weighted_preds = tf.multiply(weights1*weights2, pred_value)
        ctx_pred = tf.segment_max(weighted_preds, self.rel_context_inds)

        # object propagation
        obj_gather = tf.gather(o_feature, self.obj_context_inds)
        obj = self.pad_and_gather(o_feature, self.obj_context_o)
        pred = self.pad_and_gather(p_feature, self.obj_context_p)
        obj_ctx = tf.concat([obj, pred], axis=1)
        ctxs = tf.concat([obj_gather, obj_ctx], axis=1)
        # attention
        net = ctxs
        # net = slim.fully_connected(ctxs, 512)
        # net = slim.fully_connected(net, 512)
        weights = slim.fully_connected(net, 1)
        weighted_objs = tf.multiply(weights, obj_ctx)
        ctx_obj = tf.segment_sum(weighted_objs, self.obj_context_inds)

        return ctx_pred, ctx_obj

    def _vert_rnn_forward(self, vert_in, reuse=False):
        with tf.variable_scope('vert_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()
            (vert_out, self.vert_state) = self.vert_rnn(vert_in, self.vert_state)
        return vert_out

    def _edge_rnn_forward(self, edge_in, reuse=False):
        with tf.variable_scope('edge_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()
            (edge_out, self.edge_state) = self.edge_rnn(edge_in, self.edge_state)
        return edge_out
