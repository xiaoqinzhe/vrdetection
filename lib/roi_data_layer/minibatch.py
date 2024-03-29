# --------------------------------------------------------
# Adapted from Faster R-CNN (https://github.com/rbgirshick/py-faster-rcnn)
# Written by Danfei Xu
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
#from datasets.viz import viz_scene_graph
from roi_data_layer import data_utils
from IPython import embed
from utils.timer import Timer

def get_minibatch(roidb, num_classes, imdb):
    """Given a mini batch of roidb, construct a data blob from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    # assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    #     'num_images ({}) must divide BATCH_SIZE ({})'. \
    #     format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
    # print(cfg.TRAIN.BATCH_SIZE, num_images, rois_per_image,cfg.TRAIN.FG_FRACTION,  fg_rois_per_image)
    im_timer = Timer()
    im_timer.tic()
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    im_timer.toc()

    blobs = {'ims': im_blob}

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0), dtype=np.float32)
    rels_blob = np.zeros((0, 3), dtype=np.int32)
    rel_spt_blob = np.zeros((0), dtype=np.int8)
    fb_labels_blob = np.zeros((0), dtype=np.int32)
    rel_prior = imdb.prior
    prior_blob = np.zeros((0, rel_prior.shape[1]+1), dtype=np.float32)

    bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    all_overlaps = []

    box_idx_offset = 0

    d_timer = Timer()
    d_timer.tic()
    for im_i in range(num_images):
        spts = np.zeros((0), dtype=np.int8)
        # sample graph
        if not cfg.TRAIN.USE_SAMPLE_GRAPH:
            roi_inds, rels, spts = _sample_all_gt(roidb[im_i])
        else:
            fg_rois_per_image = 64
            rois_per_image = 2*fg_rois_per_image
            num_neg_rels = 64
            roi_inds, rels = _sample_graph(roidb[im_i],
                                        fg_rois_per_image,
                                        rois_per_image,
                                        num_neg_rels=num_neg_rels)
            # print(cfg.TRAIN.NUM_NEG_RELS)
        # print("sample roi = %i"%len(roi_inds), "sample rel = %i"%len(rels))

        # print(roi_inds)
        # print(rels)

        if roi_inds.size == 0:
            continue

        # gather all samples based on the sampled graph
        rels, labels, overlaps, im_rois, bbox_targets, bbox_inside_weights, fg_bg_labels =\
            _gather_samples(roidb[im_i], roi_inds, rels, num_classes)

        # p = data_utils.get_priors(rels, labels, num_classes, rel_prior)
        # prior_blob = np.vstack((prior_blob, p))

        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])

        batch_ind = im_i * np.ones((rois.shape[0], 1)) #im id for roi_pooling
        rois_blob_this_image = np.hstack((batch_ind, rois))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.hstack((labels_blob, labels))
        fb_labels_blob = np.hstack((fb_labels_blob, fg_bg_labels))
        bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
        bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
        all_overlaps = np.hstack((all_overlaps, overlaps))

        # offset the relationship reference idx the number of previously
        # added box
        if rels.size > 0:
            rels_offset = rels.copy()
            rels_offset[:, :2] += box_idx_offset
            rels_blob = np.vstack([rels_blob, rels_offset])
            rel_spt_blob = np.hstack((rel_spt_blob, spts))
        box_idx_offset += rois.shape[0]

        #viz_inds = np.where(overlaps == 1)[0] # ground truth
        #viz_inds = npr.choice(np.arange(rois.shape[0]), size=50, replace=False) # random sample
        #viz_inds = np.where(overlaps > cfg.TRAIN.FG_THRESH)[0]  # foreground
        #viz_scene_graph(im_blob[im_i], rois, labels, viz_inds, rels)

    if len(rois_blob) == 0 or len(rels_blob) == 0:
        return None

    blobs['rois'] = rois_blob.copy()
    blobs['labels'] = labels_blob.copy().astype(np.int32)
    blobs['relations'] = rels_blob[:,:2].copy().astype(np.int32)
    blobs['predicates'] = rels_blob[:,2].copy().astype(np.int32)
    blobs['bbox_targets'] = bbox_targets_blob.copy()
    blobs['bbox_inside_weights'] = bbox_inside_blob.copy()
    if cfg.TRAIN.USE_RPN_DB and cfg.TRAIN.USE_FG_BG:
        blobs['fg_labels'] = fb_labels_blob.copy()
        # print(blobs['fg_labels'])
    # blobs['prior'] = prior_blob.copy()
    #     np.array(bbox_inside_blob > 0).astype(np.float32).copy()


    num_roi = rois_blob.shape[0]
    num_rel = rels_blob.shape[0]
    blobs['rel_rois'] = data_utils.compute_rel_rois(num_rel,
                                                    rois_blob,
                                                    rels_blob)

    # spatial data
    # blobs['rel_spts'] = rel_spt_blob

    d_timer.toc()
    graph_dict = data_utils.create_graph_data(num_roi, num_rel, rels_blob[:, :2])

    for k in graph_dict:
        blobs[k] = graph_dict[k]

    # graph arch weights
    if True:
        obj_m, rel_m = data_utils.cal_graph_matrix(num_roi, num_rel, rels_blob)
        blobs['obj_matrix'] = obj_m
        blobs['rel_matrix'] = rel_m

    # relationship weighted
    blobs['rel_weight_labels'], blobs['rel_weight_rois'] = data_utils.cal_rel_weights(im_blob, rels_blob)

    # relationship triple ranking
    blobs['rel_triple_inds'], blobs['rel_triple_labels'] = data_utils.cal_rel_triples(rels_blob, cfg.TRAIN.NUM_SAMPLE_PAIRS)

    # show data
    # print("num_roi: ", num_roi)
    # print("num_rel", num_rel)
    # print("rois", blobs['rois'])
    # print("labels", blobs['labels'])
    # print("relations", blobs['relations'])
    # print("predicates", blobs['predicates'])
    # print("rel_rois", blobs['rel_rois'])
    # print("obj_context_o", blobs['obj_context_o'])
    # print("obj_context_p", blobs['obj_context_p'])
    # print("obj_context_inds", blobs['obj_context_inds'])
    # print("rel_context", blobs['rel_context'])
    # print("rel_context_inds", blobs['rel_context_inds'])
    # print("obj_matrix", blobs['obj_matrix'])
    # print("rel_matrix", blobs['rel_matrix'])

    return blobs

def _gather_samples(roidb, roi_inds, rels, num_classes):
    """
    join all samples and produce sampled items
    """
    rois = roidb['boxes']
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']

    # decide bg rois
    bg_inds = np.where(overlaps < cfg.TRAIN.FG_THRESH)[0]
    fg_bg_labels = np.ones(len(rois), np.int32)
    fg_bg_labels[bg_inds] = 0
    fg_bg_labels = fg_bg_labels[roi_inds]

    labels = labels.copy()
    # labels[bg_inds] = 0
    labels = labels[roi_inds]
    # print('num bg = %i' % np.where(labels==0)[0].shape[0])

    # rois and bbox targets
    overlaps = overlaps[roi_inds]
    rois = rois[roi_inds]

    # convert rel index
    roi_ind_map = {}
    for i, roi_i in enumerate(roi_inds):
        roi_ind_map[roi_i] = i
    for i, rel in enumerate(rels):
        rels[i] = [roi_ind_map[rel[0]], roi_ind_map[rel[1]], rel[2]]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
        roidb['bbox_targets'][roi_inds, :], num_classes)

    return rels, labels, overlaps, rois, bbox_targets, bbox_inside_weights, fg_bg_labels

def _sample_all_gt(roidb):
    # return all ground-truth roidb inds and rel
    return np.where(roidb['max_overlaps'] == 1)[0], roidb['gt_relations'], roidb['gt_spatial']

def _sample_graph(roidb, num_fg_rois, num_rois, num_neg_rels=128):
    """
    Sample a graph from the foreground rois of an image

    roidb: roidb of an image
    rois_per_image: maximum number of rois per image
    """
    DEBUG = False
    num_pos_rel = 64
    num_neg_rel_rate = 2
    num_fg_rois = 64
    num_bg_roi_rate = 2

    gt_rels = roidb['gt_relations']
    # index of assigned gt box for foreground boxes
    fg_gt_ind_assignments = roidb['fg_gt_ind_assignments']

    # find all fg proposals that are mapped to a gt
    gt_to_fg_roi_inds = {}
    all_fg_roi_inds = []
    for ind, gt_ind in fg_gt_ind_assignments.items():
        if gt_ind not in gt_to_fg_roi_inds:
            gt_to_fg_roi_inds[gt_ind] = []
        gt_to_fg_roi_inds[gt_ind].append(ind)
        all_fg_roi_inds.append(ind)

    # print('gt rois = %i' % np.where(roidb['max_overlaps']==1)[0].shape[0])
    # print('assigned gt = %i' % len(gt_to_fg_roi_inds.keys()))
    # dedup the roi inds
    all_fg_roi_inds = np.array(list(set(all_fg_roi_inds)))

    # find all valid relations in fg objects
    # count = np.zeros((71), dtype=np.int32)
    pos_rels = []
    gtrel_to_rels = []
    for rel in gt_rels:
        gtrel_to_rels.append([])
        for sub_i in gt_to_fg_roi_inds[rel[0]]:
            for obj_i in gt_to_fg_roi_inds[rel[1]]:
                pos_rels.append([sub_i, obj_i, rel[2]])
                gtrel_to_rels[-1].append([sub_i, obj_i, rel[2]])
                # count[rel[2]]+=1

    if DEBUG:
        print(num_fg_rois, num_rois, num_neg_rels)
        print(fg_gt_ind_assignments)
        print(gt_to_fg_roi_inds)
        print(gt_rels)
        print('num fg rois = %i' % all_fg_roi_inds.shape[0])
        print('num pos rels = %i' % len(pos_rels))

    rels = []
    rels_inds = []
    roi_inds = []

    if len(pos_rels) > 0:
        # de-duplicate the relations
        _, indices = np.unique(["{} {}".format(i, j) for i,j,k in pos_rels], return_index=True)
        pos_rels = np.array(pos_rels)[indices, :]

        # random sample

        if len(pos_rels) > num_pos_rel:
            np.random.shuffle(pos_rels)
            pos_rels = pos_rels[:num_pos_rel]
        elif len(pos_rels) < num_pos_rel:
            sample_inds = np.random.random_integers(0, len(pos_rels)-1, num_pos_rel-len(pos_rels))
            pos_rels = np.vstack([pos_rels, pos_rels[sample_inds]])

        # construct graph based on valid relations
        for rel in pos_rels:
            roi_inds += rel[:2].tolist()
            roi_inds = list(set(roi_inds)) # keep roi inds unique
            rels.append(rel)
            rels_inds.append(rel[:2].tolist())

            # if len(roi_inds) >= num_fg_rois:
            #
            #     break

    if DEBUG:
        print('sampled pos rels = %i' % len(rels))
        print('sampled fg rois = %i' % len(roi_inds))

    roi_candidates = np.setdiff1d(all_fg_roi_inds, roi_inds)
    num_rois_to_sample = min(num_fg_rois - len(roi_inds), len(roi_candidates))
    # if not enough rois, sample fg rois
    if num_rois_to_sample > 0:
        roi_sample = npr.choice(roi_candidates, size=num_rois_to_sample, replace=False)
        roi_inds = np.hstack([roi_inds, roi_sample])
        if DEBUG: print('sampled extra fg rois = %i' % num_rois_to_sample)

    """ sample background relations """
    sample_rels = []
    sample_rels_inds = []
    for i in roi_inds:
        for j in roi_inds:
            if i != j and [i, j] not in rels_inds:
                sample_rels.append([i,j,0])
                sample_rels_inds.append([i,j])

    if len(sample_rels) > 0:
        # randomly sample negative edges to prevent no edges
        num_neg_rels = np.minimum(len(sample_rels), num_neg_rels)
        if cfg.TRAIN.USE_RPN_DB:
            num_neg_rels = np.minimum(len(sample_rels), int(len(rels)*num_neg_rel_rate))
        inds = npr.choice(np.arange(len(sample_rels)), size=num_neg_rels, replace=False)
        rels += [sample_rels[i] for i in inds]
        rels_inds += [sample_rels_inds[i] for i in inds]
        if DEBUG: print('sampled neg relationships = %i/%i' % (num_neg_rels, len(sample_rels)))

    if cfg.TRAIN.USE_RPN_DB and cfg.TRAIN.USE_FG_BG:
        """ sample background relations """
        num_rois_to_sample = np.minimum(len(roidb['boxes']) - len(roi_inds), num_bg_roi_rate*len(roi_inds))
        if num_rois_to_sample > 0:
            bg_roi_inds = _sample_bg_rois(roidb, num_rois_to_sample)
            if DEBUG: print('sampled bg rois = %i' % len(bg_roi_inds))
            roi_inds = np.hstack([roi_inds, bg_roi_inds])

    roi_inds = np.array(roi_inds).astype(np.int64)
    if DEBUG:
        print("final sampled relations", len(rels), rels)
        print("final sampled rois", len(roi_inds), roi_inds)

    # print('total sampled rois = %i' % roi_inds.shape[0])
    # print('total sampled rels = %i' % len(rels))
    return roi_inds.astype(np.int64), np.array(rels).astype(np.int64)

def _sample_bg_rois(roidb, num_bg_rois):
    """
    Sample rois from background
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']

    # bg_inds = np.where(((overlaps < cfg.TRAIN.BG_THRESH_HI) &
    #                    (overlaps >= cfg.TRAIN.BG_THRESH_LO)) |
    #                    (labels == 0))[0]
    bg_inds = np.where(((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                        (overlaps >= cfg.TRAIN.BG_THRESH_LO)))[0]
    bg_rois_per_this_image = np.minimum(num_bg_rois, bg_inds.size)
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    return bg_inds

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = roidb[i]['image']() # use image getter
        if im is None:
            raise Exception('image is None')

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind].astype(np.int64)
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights
