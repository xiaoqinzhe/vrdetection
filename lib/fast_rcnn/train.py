# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

"""
Train a scene graph generation network
"""

import tensorflow as tf
import numpy as np
import os

from fast_rcnn.config import cfg
from networks.factory import get_network
from networks import losses
from roi_data_layer.data_runner import DataRunnerMP
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
from tensorflow.python import pywrap_tensorflow

class Trainer(object):

    def __init__(self, sess, net_name, imdb, roidb, output_dir, tf_log, pretrained_model=None, val_roidb=None):
        """Initialize the SolverWrapper."""
        self.net_name = net_name
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.tf_log = tf_log
        self.pretrained_model = pretrained_model
        self.bbox_means = np.zeros((self.imdb.num_classes, 4))
        self.bbox_stds = np.ones((self.imdb.num_classes, 4))
        self.val_roidb = val_roidb
        self.if_val = val_roidb is not None
        self.VAL_FREQ = 50
        self.VAL_NUM = 1
        self.VAL_BATCHES = 2
        self.init_conv = True
        self.basenet=cfg.BASENET
        self.basenet_iter=cfg.BASENET_WEIGHT_ITER
        self.pretrained_model = 'checkpoints/vrd/multinet_6_fine/pre_trained/weights_49999.ckpt'
        if self.init_conv:
            # if cfg.MODEL_PARAMS['stop_gradient']:
            self.pretrained_model = 'tf_faster_rcnn/output/{}/{}_train/default/{}_faster_rcnn_iter_{}.ckpt'.format(self.basenet, cfg.DATASET, self.basenet, self.basenet_iter)
            # self.pretrained_model = 'tf_faster_rcnn/data/imagenet_weights/imagenet_vgg16.ckpt'

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
            print('Loaded precomputer bbox target distribution from %s' % \
                  cfg.TRAIN.BBOX_TARGET_NORMALIZATION_FILE)
            bbox_dist = np.load(cfg.TRAIN.BBOX_TARGET_NORMALIZATION_FILE).item()
            self.bbox_means = bbox_dist['means']
            self.bbox_stds = bbox_dist['stds']

        print('done')


    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers and cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()
            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds.ravel(), (weights_shape[0],1))))
            sess.run(biases.assign(orig_1 * self.bbox_stds.ravel() + self.bbox_means.ravel()))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename = os.path.join(self.output_dir, 'weights_%i.ckpt' % iter)

        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers and cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
            # restore net to original state
            sess.run(weights.assign(orig_0))
            sess.run(biases.assign(orig_1))

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def load_pretrained_models(self, sess):
        if self.pretrained_model is not None:
            print(('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model))
            if self.pretrained_model.endswith('.npy'):
                self.net.load(self.pretrained_model, sess, load_fc=True)
            elif self.pretrained_model.endswith('.ckpt'):
                if self.init_conv:
                    # load pre_trained model's weights
                    var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
                    if "imagenet" in self.pretrained_model:
                        variables_to_restore = self.net.get_variables_to_restore_imagenet(tf.global_variables(), var_keep_dic)
                        restorer = tf.train.Saver(variables_to_restore)
                        restorer.restore(sess, self.pretrained_model)
                        print('Loaded.')
                        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
                        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
                        # fully connected weights
                        self.net.fix_variables_imagenet(sess, self.pretrained_model)
                        print('Fixed.')
                    else:
                        var_list = []
                        for var in tf.global_variables(self.net._scope):
                            if var.name.split(':')[0] in var_keep_dic:
                                var_list.append(var)
                        restorer = tf.train.Saver(var_list)
                        restorer.restore(sess, self.pretrained_model)
                else:
                    # load old trained models
                    var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
                    var_list = []
                    for var in tf.global_variables():
                        if var.name.split(':')[0] in var_keep_dic:
                            if var.name.split(':')[0] == 'learning_rate': continue
                            if 'Momentum' in var.name.split(':')[0]: continue
                            # if 'rel_score' in var.name.split(':')[0]: continue
                            var_list.append(var)
                    print(var_list)
                    restorer = tf.train.Saver(var_list)
                    restorer.restore(sess, self.pretrained_model)
            else:
                print('Unsupported pretrained weights format')
                raise Exception

    def get_data_runner(self, sess, input_pls, data_layer, capacity=30):

        def data_generator():
            while True:
                yield data_layer.next_batch()

        def task_generator():
            while True:
                yield data_layer._get_next_minibatch_inds()

        task_func = data_layer._get_next_minibatch
        data_runner = DataRunnerMP(task_func, task_generator, input_pls, capacity=capacity)

        return data_runner


    def train_model(self, sess, max_iters):
        input_pls = get_network(self.net_name).inputs(self.imdb.num_classes, self.imdb.prior.shape[1], self.imdb.embedding_size, is_training=True)
        # input_pls = {
        #     'ims': tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]),
        #     'rois': tf.placeholder(dtype=tf.float32, shape=[None, 5]),
        #     'rel_rois': tf.placeholder(dtype=tf.float32, shape=[None, 5]),
        #     'labels': tf.placeholder(dtype=tf.int32, shape=[None]),
        #     # 'bboxes': tf.placeholder(dtype=tf.float32, shape=[None, 4]),
        #     'relations': tf.placeholder(dtype=tf.int32, shape=[None, 2]),
        #     'predicates': tf.placeholder(dtype=tf.int32, shape=[None]),
        #     'rel_spts': tf.placeholder(dtype=tf.int32, shape=[None]),
        #     'bbox_targets': tf.placeholder(dtype=tf.float32, shape=[None, 4 * self.imdb.num_classes]),
        #     'bbox_inside_weights': tf.placeholder(dtype=tf.float32, shape=[None, 4 * self.imdb.num_classes]),
        #     'num_roi': tf.placeholder(dtype=tf.int32, shape=[]),  # number of rois per batch
        #     'num_rel': tf.placeholder(dtype=tf.int32, shape=[]),  # number of relationships per batch
        #     'obj_context_o': tf.placeholder(dtype=tf.int32, shape=[None]),
        #     'obj_context_p': tf.placeholder(dtype=tf.int32, shape=[None]),
        #     'obj_context_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
        #     'rel_context': tf.placeholder(dtype=tf.int32, shape=[None]),
        #     'rel_context_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
        #     'obj_embedding': tf.placeholder(dtype=tf.float32, shape=[self.imdb.num_classes, self.imdb.embedding_size]),
        #     'obj_matrix': tf.placeholder(dtype=tf.float32, shape=[None, None]),
        #     'rel_matrix': tf.placeholder(dtype=tf.float32, shape=[None, None]),
        #     'rel_weight_labels': tf.placeholder(dtype=tf.int32, shape=[None]),
        #     'rel_weight_rois': tf.placeholder(dtype=tf.float32, shape=[None, 5]),
        #     'rel_triple_inds': tf.placeholder(dtype=tf.int32, shape=[None, 2]),
        #     'rel_triple_labels': tf.placeholder(dtype=tf.int32, shape=[None, 1]),
        # }
        """Network training loop."""
        data_layer = RoIDataLayer(self.imdb, self.roidb, self.imdb.num_classes, self.bbox_means, self.bbox_stds)

        # a multi-process data runner
        data_runner = self.get_data_runner(sess, input_pls, data_layer)

        if self.if_val:
            val_data_layer = RoIDataLayer(self.imdb, self.val_roidb, self.imdb.num_classes, self.bbox_means, self.bbox_stds,
                                          num_batches=self.VAL_BATCHES)
            val_data_runner = self.get_data_runner(sess, input_pls, val_data_layer, capacity=self.VAL_NUM*self.VAL_BATCHES)

        inputs= data_runner.get_inputs()

        inputs['num_classes'] = self.imdb.num_classes
        inputs['num_predicates'] = self.imdb.num_predicates
        inputs['num_spatials'] = self.imdb.num_spatials
        inputs['n_iter'] = cfg.TRAIN.INFERENCE_ITER
        inputs['is_training'] = True

        # data_runner.start_threads(sess, n_threads=10)
        data_runner.start_processes(sess, n_processes=3)
        if self.if_val:
            val_data_runner.start_processes(sess, n_processes=2)

        print("classes = %i"%self.imdb.num_classes, "predicates = %i"%self.imdb.num_predicates)

        ## net settings
        for key in cfg.MODEL_PARAMS:
            inputs[key] = cfg.MODEL_PARAMS[key]
        inputs['basenet']=self.basenet
        self.net = get_network(self.net_name)(inputs)
        self.net.setup()

        # get network-defined losses, metrics
        ops = self.net.losses()
        self.net.metrics(ops)

        ## val ops
        val_ops = {}
        for k in ops: val_ops[k] = ops[k]

        # multitask loss
        loss_list = [ops[k] for k in ops if k.startswith('loss')]
        # ops['loss_total'] = tf.add_n([ops['loss_cls'], ops['loss_rel']])
        # ops['loss_total'] = losses.total_loss_and_summaries(loss_list, 'total_loss')

        # summary losses
        for k in ops:
            # print(k)
            tf.summary.scalar(k, ops[k])

        # optimizer
        lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False, name='learning_rate')
        momentum = cfg.TRAIN.MOMENTUM
        # ops['train'] = tf.train.MomentumOptimizer(lr, momentum).minimize(tf.add(ops['loss_weight'], ops['loss_triple']))
        ops['train'] = tf.train.MomentumOptimizer(lr, momentum).minimize(ops['loss_total'])
        # ops['train'] = tf.train.AdamOptimizer(lr).minimize(ops['loss_total'])
        # ops['train'] = tf.train.GradientDescentOptimizer(lr).minimize(ops['loss_total'])
        # if not self.net.stop_gradient:
        #     # different learning rate
        #     all_variables = tf.trainable_variables()
        #     finetune_variables = tf.trainable_variables(scope=self.net._scope)
        #     new_variables = []
        #     for var in all_variables:
        #         if var not in finetune_variables:
        #             new_variables.append(var)
        #     print("finetune_variables", finetune_variables)
        #     print("new_variables", new_variables)
        #     opt_finetune = tf.train.MomentumOptimizer(lr*0.1, momentum)
        #     opt_new = tf.train.MomentumOptimizer(lr, momentum)
        #     grads = tf.gradients(ops['loss_total'], finetune_variables+new_variables)
        #     op_finetune = opt_finetune.apply_gradients(zip(grads[:len(finetune_variables)], finetune_variables))
        #     op_new = opt_new.apply_gradients(zip(grads[len(finetune_variables):], new_variables))
        #     train_op = tf.group(op_finetune, op_new)
        #     ops['train'] = train_op

        # ops merge summaries
        ops_summary = dict(ops)
        ops_summary['summary'] = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.tf_log, sess.graph)
        print(self.tf_log)
        val_writer = tf.summary.FileWriter(self.tf_log+'val/')

        # intialize variables
        sess.run(tf.global_variables_initializer())
        if self.net.use_embedding:
            sess.run(self.net.embedding_init, feed_dict={inputs['obj_embedding']:self.imdb.word2vec[:self.imdb.num_classes]})
        self.saver = tf.train.Saver(max_to_keep=25)
        self.load_pretrained_models(sess)
        print("load done")

        last_snapshot_iter = -1
        timer = Timer()
        iter_timer = Timer()

        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = list(cfg.TRAIN.STEPSIZES)
        stepsizes.append(max_iters+1)
        stepsizes.reverse()
        next_stepsize = stepsizes.pop()

        # Training loop
        for iter in range(max_iters):
            # tracing training information
            if iter % 10000 == 0:
                run_metadata = tf.RunMetadata()
                feed_dict = data_runner.get_feed_batch()
                feed_dict[self.net.keep_prob] = 0.5
                _ = sess.run(ops, feed_dict=feed_dict, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, "step{}".format(iter), iter)

            iter_timer.tic()

            # learning rate decay
            # 1. simple decay
            # if (iter+1) % cfg.TRAIN.STEPSIZE == 0:
            #     sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))
            if iter == next_stepsize:
                rate *= cfg.TRAIN.GAMMA
                sess.run(tf.assign(lr, rate))
                next_stepsize = stepsizes.pop()
            # 2. cosine decay
            # if iter % 1000 ==0:
            #     new_lr_value = 0.5 * cfg.TRAIN.LEARNING_RATE * (1 + np.cos(iter*np.pi/max_iters))
            #     sess.run(tf.assign(lr, new_lr_value))

            # Make one SGD update
            feed_dict = data_runner.get_feed_batch()
            # print(feed_dict[inputs['rel_triple_labels']])
            feed_dict[self.net.keep_prob] = 0.5
            timer.tic()
            if (iter + 1) % cfg.TRAIN.SUMMARY_FREQ == 0:
                ops_value = sess.run(ops_summary, feed_dict=feed_dict)
                train_writer.add_summary(ops_value['summary'], iter)
            else:
                ops_value = sess.run(ops, feed_dict=feed_dict)

            timer.toc()

            if np.isnan(ops_value['loss_total']):
                print('nan', iter)
                for k in feed_dict:
                    print(k, feed_dict[k])
                exit()

            if iter%20 == 0:
                stats = 'iter: %d / %d, lr: %f' % (iter+1, max_iters, lr.eval())
                for k in ops_value:
                    if k.startswith('loss') or k.startswith('acc') or k.startswith('rec'):
                        stats += ', %s: %4f' % (k, ops_value[k])
                print(stats)

            # print(ops_value['rel'])

            if self.if_val and ((iter+1) % self.VAL_FREQ == 0 or (iter+1) == max_iters-1):
                val_sums = {}
                for k in val_ops: val_sums[k] = 0
                # if (iter+1) % 10000 == 0: times = 900//self.VAL_BATCHES
                # else:
                times = self.VAL_NUM
                for i in range(times):
                    feed_dict = val_data_runner.get_feed_batch()
                    feed_dict[self.net.keep_prob] = 1
                    ops_value, val_summary = sess.run([val_ops, ops_summary['summary']], feed_dict=feed_dict)
                    if i==0: val_writer.add_summary(val_summary, iter)
                    for k in val_ops: val_sums[k] += ops_value[k]
                for k in val_sums: val_sums[k] /= times
                stats = '**--validate iter--**: %d / %d, %d, lr: %f' % (iter + 1, max_iters, iter//self.VAL_FREQ, lr.eval())
                for k in val_sums:
                    stats += ', %s: %4f' % (k, val_sums[k])
                print(stats)

            iter_timer.toc()

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY_FREQ) == 0:
                print('speed: {:.3f}s / iter'.format(timer.average_time))
                print('iter speed: {:.3f}s / iter'.format(iter_timer.average_time))

            # if (iter+1) % cfg.TRAIN.SNAPSHOT_FREQ == 0:
            if (iter + 1) % 5000 == 0:
            # if (iter + 1) % 5000 == 0 or (iter > 30000 and iter % 2000 == 0):
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


def train_net(network_name, imdb, roidb, output_dir, tf_log, pretrained_model=None, max_iters=200000, val_roidb=None):
    if network_name in ["weightnet", "ranknet", 'ctxnet', 'graphnet']:
        cfg.TRAIN.USE_GRAPH_SAMPLE=True
    else: cfg.TRAIN.USE_GRAPH_SAMPLE=False
    config = tf.ConfigProto()
    config.allow_soft_placement=True
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        tf.set_random_seed(cfg.RNG_SEED)
        trainer = Trainer(sess, network_name, imdb, roidb, output_dir, tf_log, pretrained_model=pretrained_model, val_roidb=val_roidb)
        trainer.train_model(sess, max_iters)