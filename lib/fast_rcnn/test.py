# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from roi_data_layer.roidb import prepare_roidb
import roi_data_layer.data_utils as data_utils
from datasets.evaluator import SceneGraphEvaluator
from networks.factory import get_network
from utils.timer import Timer
from utils.cpu_nms import cpu_nms
import numpy as np
import scipy.ndimage
import tensorflow as tf
import os
from utils.blob import im_list_to_blob

"""
Test a scene graph generation network
"""

word2vec = []

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

        im = scipy.ndimage.interpolation.zoom(im_orig, (im_scale, im_scale, 1.0), order=1)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    # print(im_rois.shape, scales.shape, levels.shape)
    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(sess, net, inputs, im, boxes, bbox_reg, multi_iter, roidb, pred_ops, metric_ops):
    blobs, im_scales = _get_blobs(im, boxes)

    rel_pos = np.ones([boxes.shape[0], boxes.shape[0]], np.int8)*-1
    k = 0
    relations = []
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[0]):
            if i != j:
                relations.append([i, j])
                rel_pos[i][j] = k
                k += 1

    num_roi = blobs['rois'].shape[0]

    predicates = roidb['gt_relations'][:, 2]

    if cfg.TEST.METRIC_EVAL:
        if not cfg.TRAIN.USE_SAMPLE_GRAPH:
            relations = roidb['gt_relations'][:, 0:2]
        else:
            predicates = np.zeros(len(relations), np.int32)
            for rel in roidb['gt_relations']:
                predicates[rel_pos[rel[0], rel[1]]] = rel[2]

    relations = np.array(relations, dtype=np.int32)  # all possible combinations


    num_rel  = relations.shape[0]

    inputs_feed = data_utils.create_graph_data(num_roi, num_rel, relations)
    global word2vec
    inputs_feed['obj_embedding'] = word2vec

    feed_dict = {inputs['ims']: blobs['data'],
                 inputs['rois']: blobs['rois'],
                 inputs['relations']: relations,
                 inputs['rel_spts']: roidb['gt_spatial'],
                 inputs['labels']: roidb['gt_classes'],
                 inputs['predicates']: predicates,
                 net.keep_prob: 1}

    for k in inputs_feed:
        feed_dict[inputs[k]] = inputs_feed[k]

    # compute relation rois
    feed_dict[inputs['rel_rois']] = \
        data_utils.compute_rel_rois(num_rel, blobs['rois'], relations)

    ops_value, metrics = sess.run([pred_ops, metric_ops], feed_dict=feed_dict)

    if ops_value is None: return None, metrics

    # for key in ops_value:
    #     if key.startswith('acc'):
    #         print(key, ops_value[key])

    out_dict = {}
    for mi in multi_iter:
        rel_probs = None
        rel_probs_flat = ops_value['rel_probs'][mi]
        # rel_vis_probs_flat = ops_value['rel_probs_vis'][mi]
        # rel_probs_flat = np.mean([rel_probs_flat, rel_vis_probs_flat], axis=0)
        rel_probs = np.zeros([num_roi, num_roi, rel_probs_flat.shape[1]])
        for i, rel in enumerate(relations):
            rel_probs[rel[0], rel[1], :] = rel_probs_flat[i, :]
        if net.if_pred_cls:
            cls_probs = ops_value['cls_probs'][mi]
        else:
            cls_probs = np.zeros((boxes.shape[0], inputs['num_classes']), dtype=np.int8)
            for i in range(boxes.shape[0]): cls_probs[i, roidb['gt_classes'][i]] = 1

        if bbox_reg:
            # Apply bounding-box regression deltas
            pred_boxes = bbox_transform_inv(boxes, ops_value['bbox_deltas'][mi])
            pred_boxes = clip_boxes(pred_boxes, im.shape)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, cls_probs.shape[1]))

        out_dict[mi] = {'scores': cls_probs.copy(),
                        'boxes': pred_boxes.copy(),
                        'relations': rel_probs.copy()}

    return out_dict, metrics

def non_gt_rois(roidb):
    overlaps = roidb['max_overlaps']
    gt_inds = np.where(overlaps == 1)[0]
    non_gt_inds = np.setdiff1d(np.arange(overlaps.shape[0]), gt_inds)
    rois = roidb['boxes'][non_gt_inds]
    scores = roidb['roi_scores'][non_gt_inds]
    return rois, scores

def gt_rois(roidb):
    overlaps = roidb['max_overlaps']
    gt_inds = np.where(overlaps == 1)[0]
    rois = roidb['boxes'][gt_inds]
    return rois

def test_net(net_name, weight_name, imdb, mode, max_per_image=100):
    sess = tf.Session()

    # set up testing mode
    rois = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='rois')
    rel_rois = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='rel_rois')
    ims = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='ims')
    relations = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='relations')
    inputs = {'rois': rois,
              'rel_rois': rel_rois,
              'ims': ims,
              'labels': tf.placeholder(dtype=tf.int32, shape=[None]),
              'relations': tf.placeholder(dtype=tf.int32, shape=[None, 2]),
              'predicates': tf.placeholder(dtype=tf.int32, shape=[None]),
              'rel_spts': tf.placeholder(dtype=tf.int32, shape=[None]),
              'num_roi': tf.placeholder(dtype=tf.int32, shape=[]),
              'num_rel': tf.placeholder(dtype=tf.int32, shape=[]),
              'num_classes': imdb.num_classes,
              'num_predicates': imdb.num_predicates,
              'num_spatials': imdb.num_spatials,
              'obj_context_o': tf.placeholder(dtype=tf.int32, shape=[None]),
              'obj_context_p': tf.placeholder(dtype=tf.int32, shape=[None]),
              'obj_context_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              'rel_context': tf.placeholder(dtype=tf.int32, shape=[None]),
              'rel_context_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              'obj_embedding': tf.placeholder(dtype=tf.float32, shape=[None, 300]),
              #'n_iter': cfg.TEST.INFERENCE_ITER
              }

    # get network setting
    for key in cfg.MODEL_PARAMS:
        inputs[key] = cfg.MODEL_PARAMS[key]

    net = get_network(net_name)(inputs)
    net.setup()
    print ('Loading model weights from {:s}').format(weight_name)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, weight_name)

    roidb = imdb.roidb
    if cfg.TEST.USE_RPN_DB:
        imdb.add_rpn_rois(roidb, make_copy=False)
    prepare_roidb(roidb)

    global word2vec
    word2vec = imdb.word2vec

    num_images = len(imdb.image_index)

    # timers
    _t = {'im_detect' : Timer(), 'evaluate' : Timer()}

    if mode == 'all':
        eval_modes = ['pred_cls', 'sg_cls', 'sg_det']
    else:
        eval_modes = [mode]
    multi_iter = [0]
    print('Graph Inference Iteration ='),
    print(multi_iter)
    print('EVAL MODES ='),
    print(eval_modes)

    # metrics to show
    if cfg.TEST.METRIC_EVAL:
        metric_ops = net.losses()
        net.metrics(metric_ops)
    else:
        metric_ops = tf.no_op(name='no_test_metric')

    # initialize evaluator for each task
    evaluators = {}
    for m in eval_modes:
        evaluators[m] = {}
        for it in multi_iter:
            if cfg.TEST.METRIC_EVAL:
                evaluators[m][it] = SceneGraphEvaluator(imdb, mode=m, metrics=metric_ops.keys())
            else: evaluators[m][it] = SceneGraphEvaluator(imdb, mode=m)

    # rel predictions
    ops = {}
    if cfg.TEST.REL_EVAL:
        if cfg.MODEL_PARAMS['if_pred_bbox']:
            ops['bbox_deltas'] = net.bbox_pred_output(multi_iter)
        ops['rel_probs'] = net.rel_pred_output(multi_iter)
        # ops['rel_probs'] = net.rel_pred_output('_vis')
        if net.if_pred_cls:
            ops['cls_probs'] = net.cls_pred_output(multi_iter)
    else:
        ops = tf.no_op(name="no_pred_sg")



    for im_i in xrange(num_images):

        im = imdb.im_getter(im_i)

        for mode in eval_modes:
            bbox_reg = True
            if mode == 'pred_cls' or mode == 'sg_cls':
                # use ground truth object locations
                bbox_reg = False
                box_proposals = gt_rois(roidb[im_i])
            else:
                # use RPN-proposed object locations
                box_proposals, roi_scores = non_gt_rois(roidb[im_i])
                roi_scores = np.expand_dims(roi_scores, axis=1)
                nms_keep = cpu_nms(np.hstack((box_proposals, roi_scores)).astype(np.float32),
                            cfg.TEST.PROPOSAL_NMS)
                nms_keep = np.array(nms_keep)
                num_proposal = min(cfg.TEST.NUM_PROPOSALS, nms_keep.shape[0])
                keep = nms_keep[:num_proposal]
                box_proposals = box_proposals[keep, :]


            if box_proposals.size == 0 or box_proposals.shape[0] < 2:
                # continue if no graph
                print("image %d in mode %s has no or one box proposal"%(im_i, mode))
                continue
            if len(roidb[im_i]['gt_relations']) == 0:
                print("image %d in mode %s has no relation" % (im_i, mode))
                continue

            _t['im_detect'].tic()
            out_dict, metrics_v = im_detect(sess, net, inputs, im, box_proposals,
                                 bbox_reg, multi_iter, roidb[im_i], ops, metric_ops)
            _t['im_detect'].toc()
            _t['evaluate'].tic()

            for iter_n in multi_iter:
                if out_dict is not None:
                    sg_entry = out_dict[iter_n]
                    evaluators[mode][iter_n].evaluate_scene_graph_entry(sg_entry, im_i, iou_thresh=0.5)
                if metrics_v is not None: evaluators[mode][iter_n].add_metrics(metrics_v)
            _t['evaluate'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(im_i + 1, num_images, _t['im_detect'].average_time,
                      _t['evaluate'].average_time)

    # print out evaluation results
    for mode in eval_modes:
        for iter_n in multi_iter:
            evaluators[mode][iter_n].print_stats()
