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
from datasets.factory import get_detections_filename
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

def get_oo_id(i, j, num_class):
    return i*num_class + j

prior = None

def get_prior(filename):
    global prior
    if prior is None:
        import pickle
        prior = pickle.load(open(filename))
    return prior

o2o_prior = None

def get_o2o_prior(filename):
    global o2o_prior
    if o2o_prior is None:
        import pickle
        o2o_prior = pickle.load(open(filename))
    return o2o_prior

detections = None
def get_detections(im_i):
    global detections
    if detections is None:
        filename = get_detections_filename()
        import pickle
        detections = pickle.load(open(filename, 'rb'))
    return [detections[j][im_i] for j in range(len(detections))]

def im_detect(sess, net, inputs, im, im_i, boxes, bbox_reg, multi_iter, roidb, pred_ops, metric_ops, mode, cls_preds, cls_scores):
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
    predicates = np.zeros(len(relations), np.int32)

    if cfg.TEST.METRIC_EVAL:
        if not cfg.TRAIN.USE_SAMPLE_GRAPH:
            relations = roidb['gt_relations'][:, 0:2]
            predicates = roidb['gt_relations'][:,2]
        else:
            predicates = np.zeros(len(relations), np.int32)
            for rel in roidb['gt_relations']:
                predicates[rel_pos[rel[0], rel[1]]] = rel[2]

    relations = np.array(relations, dtype=np.int32)  # all possible combinations


    num_rel  = relations.shape[0]

    inputs_feed = data_utils.create_graph_data(num_roi, num_rel, relations)

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

    feed_dict[inputs['obj_matrix']], feed_dict[inputs['rel_matrix']] = \
        data_utils.cal_graph_matrix(num_roi, num_rel, relations)

    feed_dict[inputs['rel_weight_labels']], feed_dict[inputs['rel_weight_rois']] = \
        data_utils.cal_rel_weights(blobs['data'], np.hstack((relations, np.expand_dims(predicates, axis=1))))

    ops_value, metrics = sess.run([pred_ops, metric_ops], feed_dict=feed_dict)

    if ops_value is None: return None, metrics

    # for key in ops_value:
    #     if key.startswith('acc'):
    #         print(key, ops_value[key])

    out_dict = {}
    for mi in multi_iter:
        rel_probs = None
        rel_probs_flat = ops_value['rel_probs'][mi]
        if 'rel_weighted_probs' in ops_value:
            rel_probs_flat = ops_value['rel_weighted_probs']
        # rel_vis_probs_flat = ops_value['rel_probs_vis'][mi]
        # rel_probs_flat = np.max([rel_probs_flat, rel_vis_probs_flat], axis=0)
        rel_probs = np.zeros([num_roi, num_roi, rel_probs_flat.shape[1]])
        for i, rel in enumerate(relations):
            rel_probs[rel[0], rel[1], :] = rel_probs_flat[i, :]

            # using prior
            if mode == 'pred_cls':
                labels = roidb['gt_classes']
            elif mode == 'sg_det':
                labels = cls_preds
            prior = get_prior("./data/vrd/lang_prior.pickle")
            # prior = get_prior("./data/vrd/dataset_prior.pickle")
            o2o_prior = get_o2o_prior('./data/vrd/lang_prior_o2o.pickle')
            num_class = inputs['num_classes']

            prior_predicate = prior[get_oo_id(labels[rel[0]]-1, labels[rel[1]]-1, num_class-1), :]
            o2o_prior = o2o_prior[labels[rel[0]]-1, labels[rel[1]]-1]

            # rel_probs[rel[0], rel[1], :] = np.max((rel_probs_flat[i, :], prior_predicate), axis=0)
            if cfg.TRAIN.USE_SAMPLE_GRAPH:
                rel_probs[rel[0], rel[1], 1:] = rel_probs_flat[i, 1:] * prior_predicate
            else:
                rel_probs[rel[0], rel[1], :] = rel_probs_flat[i, :] * prior_predicate

            # use prior only
            # if cfg.TRAIN.USE_SAMPLE_GRAPH:
            #     rel_probs[rel[0], rel[1], 1:] = prior_predicate
            #     # rel_probs[rel[0], rel[1], 1:] = prior_predicate * ops_value['rel_weight_prob'][i]
            # else:
            #     rel_probs[rel[0], rel[1], :] = prior_predicate

        if net.if_pred_cls:
            cls_probs = ops_value['cls_probs'][mi]
        else:
            cls_probs = np.zeros((boxes.shape[0], inputs['num_classes']), dtype=np.float32)
            if mode == 'pred_cls':
                for i in range(boxes.shape[0]): cls_probs[i, roidb['gt_classes'][i]] = 1.0
            else: cls_probs = cls_scores

        out_dict[mi] = {'scores': cls_probs.copy(),
                        'boxes': boxes.copy(),
                        'relations': rel_probs.copy()}

        # cal rels show
        # if cfg.TRAIN.USE_SAMPLE_GRAPH:
        #     predicts = np.argmax(rel_probs_flat[:, 1:], axis=1) + 1
        #     show_inds = np.where(predicates)
        # else:
        #     predicts = np.argmax(rel_probs_flat, axis=1)
        #     show_inds = np.arange(len(predicates))
        # labels = roidb['gt_classes']
        # predicts = np.expand_dims(predicts[show_inds], axis=1)
        # rels = labels[relations[show_inds]]
        # gt_rels = np.expand_dims(predicates[show_inds], axis=1)
        # # print(predicts, rels, gt_rels)
        # rels_show = np.hstack((rels, gt_rels, predicts))
        # out_dict[mi]['rels_show'] = rels_show

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
              'obj_embedding': tf.placeholder(dtype=tf.float32, shape=[imdb.num_classes, cfg.WORD2VEC_SIZE]),
              'obj_matrix': tf.placeholder(dtype=tf.float32, shape=[None, None]),
              'rel_matrix': tf.placeholder(dtype=tf.float32, shape=[None, None]),
              'rel_weight_labels': tf.placeholder(dtype=tf.int32, shape=[None]),
              'rel_weight_rois': tf.placeholder(dtype=tf.float32, shape=[None, 5]),
              #'n_iter': cfg.TEST.INFERENCE_ITER
              }

    # get network setting
    for key in cfg.MODEL_PARAMS:
        inputs[key] = cfg.MODEL_PARAMS[key]
    inputs['basenet']=cfg.BASENET
    inputs['is_training'] = False
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

    if mode == 'sg_det':
        cfg.TEST.METRIC_EVAL = False

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
        if 'rel_weighted_prob' in net.layers:
            print("using weighted rel prob!!!!!!!!")
            ops['rel_weighted_probs'] = net.layers['rel_weighted_prob']
            ops['rel_weight_prob'] = net.layers['rel_weight_prob']
        # ops['rel_probs'] = net.rel_pred_output('_vis')
        # ops['rel_probs_vis'] = net.rel_pred_output('_vis')
        if net.if_pred_cls:
            ops['cls_probs'] = net.cls_pred_output(multi_iter)
    else:
        ops = tf.no_op(name="no_pred_sg")



    for im_i in xrange(num_images):

        im = imdb.im_getter(im_i)

        for mode in eval_modes:
            bbox_reg = False
            box_proposals = []
            cls_preds = []
            cls_scores = []
            if mode == 'pred_cls' or mode == 'sg_cls':
                # use ground truth object locations
                bbox_reg = False
                box_proposals = gt_rois(roidb[im_i])
            else:
                # use RPN-proposed object locations
                # box_proposals, roi_scores = non_gt_rois(roidb[im_i])
                # roi_scores = np.expand_dims(roi_scores, axis=1)
                # nms_keep = cpu_nms(np.hstack((box_proposals, roi_scores)).astype(np.float32),
                #             cfg.TEST.PROPOSAL_NMS)
                # nms_keep = np.array(nms_keep)
                # num_proposal = min(cfg.TEST.NUM_PROPOSALS, nms_keep.shape[0])
                # keep = nms_keep[:num_proposal]
                # box_proposals = box_proposals[keep, :]
                detected_res = get_detections(im_i)
                for j in range(1, imdb.num_classes):
                    if(len(detected_res[j])==0): continue
                    # print(detected_res[j], [detected_res[j][k][:4] for k in range(len(detected_res[j]))])
                    box_proposals.extend([detected_res[j][k][:4] for k in range(len(detected_res[j]))])
                    cls_preds.extend([j for _ in range(len(detected_res[j]))])
                    cls_scores.extend([detected_res[j][k][4] for k in range(len(detected_res[j]))])
                box_proposals = np.array(box_proposals)
                cls_preds = np.array(cls_preds)
                cls_scores = np.array(cls_scores)

                # filter box
                filter_gt_box = False
                max_boxes = 20
                score_thresh = 0.
                if score_thresh>0.0:
                    th_inds = np.where(cls_scores>score_thresh)
                    box_proposals = box_proposals[th_inds]
                    cls_preds = cls_preds[th_inds]
                    cls_scores = cls_scores[th_inds]
                    print("after filtered by score_thresh", len(box_proposals))
                if max_boxes>0 and len(cls_preds)>max_boxes:
                    inds = np.argsort(-cls_scores)
                    th_inds = inds[:max_boxes]
                    box_proposals = box_proposals[th_inds]
                    cls_preds = cls_preds[th_inds]
                    cls_scores = cls_scores[th_inds]
                if filter_gt_box:
                    max_iou = 0.5
                    from utils.cython_bbox import bbox_overlaps
                    overlaps = bbox_overlaps(box_proposals.astype(np.float), gt_rois(roidb[im_i]).astype(np.float))
                    maxes = np.max(overlaps, axis=1)
                    inds = np.where(maxes>max_iou)
                    box_proposals = box_proposals[inds]
                    cls_preds = cls_preds[inds]
                    cls_scores = cls_scores[inds]
                    print("after filtered by gt box iou>0.5", len(box_proposals))

                # cls_scores = np.ones_like(cls_scores, np.float)

                print("predicted boxes have {}".format(len(box_proposals)))

            if box_proposals.size == 0 or box_proposals.shape[0] < 2:
                # continue if no graph
                print("image %d in mode %s has no or one box proposal"%(im_i, mode))
                continue
            if len(roidb[im_i]['gt_relations']) == 0:
                print("image %d in mode %s has no relation" % (im_i, mode))
                continue

            _t['im_detect'].tic()
            out_dict, metrics_v = im_detect(sess, net, inputs, im, im_i,box_proposals,
                                 bbox_reg, multi_iter, roidb[im_i], ops, metric_ops, mode, cls_preds, cls_scores)
            _t['im_detect'].toc()
            _t['evaluate'].tic()

            for iter_n in multi_iter:
                if out_dict is not None:
                    sg_entry = out_dict[iter_n]
                    if mode == 'sg_det':
                        sg_entry['boxes'] = box_proposals
                        sg_entry['cls_preds'] = cls_preds
                        sg_entry['cls_scores'] = cls_scores
                    evaluators[mode][iter_n].evaluate_scene_graph_entry(sg_entry, im_i, iou_thresh=0.5)
                    # evaluators[mode][iter_n].add_rels_to_show(sg_entry['rels_show'])
                if metrics_v is not None: evaluators[mode][iter_n].add_metrics(metrics_v)
            _t['evaluate'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(im_i + 1, num_images, _t['im_detect'].average_time,
                      _t['evaluate'].average_time)

    # print out evaluation results
    for mode in eval_modes:
        for iter_n in multi_iter:
            evaluators[mode][iter_n].print_stats()
            # evaluators[mode][iter_n].save_rels_to_show(open('./data/viz/rels_train_show.txt', 'w'), save_true_pred=True)
