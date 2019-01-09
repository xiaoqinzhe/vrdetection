import numpy as np
from fast_rcnn.config import cfg

total = 0

def eval_relation_recall(sg_entry,
                         roidb_entry,
                         result_dict,
                         mode,
                         iou_thresh, num_k=70, use_gt_rel = False):
    use_gt_rel = cfg.TEST.METRIC_EVAL
    # gt
    gt_inds = np.where(roidb_entry['max_overlaps'] == 1)[0]
    gt_boxes = roidb_entry['boxes'][gt_inds].copy().astype(float)
    num_gt_boxes = gt_boxes.shape[0]
    gt_relations = roidb_entry['gt_relations'].copy()
    gt_classes = roidb_entry['gt_classes'].copy()

    num_gt_relations = gt_relations.shape[0]
    if num_gt_relations == 0:
        return (None, None)
    gt_class_scores = np.ones(num_gt_boxes)
    gt_predicate_scores = np.ones(num_gt_relations)
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_relations[:,2],
                                             gt_relations[:,:2],
                                             gt_classes,
                                             gt_boxes,
                                             gt_predicate_scores,
                                             gt_class_scores)
    num_true_gt_rels = None
    # cal gt_rel true number
    # num_true_gt_rels = len(np.unique(gt_relations[:,:2], axis=0))
    # global total
    # if num_true_gt_rels<num_gt_relations: total += 1
    # print(num_gt_relations, num_true_gt_rels, total)

    # pred
    box_preds = sg_entry['boxes']
    num_boxes = box_preds.shape[0]
    predicate_preds = sg_entry['relations']
    class_preds = sg_entry['scores']
    predicate_preds = predicate_preds.reshape(num_boxes, num_boxes, -1)

    # no bg
    if cfg.TRAIN.USE_SAMPLE_GRAPH:
        predicate_preds = predicate_preds[:, :, 1:]

    relations = []
    predicates = []
    predicate_scores = []
    if use_gt_rel and not cfg.TRAIN.USE_SAMPLE_GRAPH:
    # if use_gt_rel:
        for rel in gt_relations:
            i, j = rel[0], rel[1]
            arg_sort = np.argsort(-predicate_preds[i][j])
            for k in range(num_k):
                relations.append([i, j])
                if cfg.TRAIN.USE_SAMPLE_GRAPH: predicates.append(arg_sort[k] + 1)
                else: predicates.append(arg_sort[k])
                predicate_scores.append(predicate_preds[i][j][arg_sort[k]])
    else:
        for i in xrange(num_boxes):
            for j in xrange(num_boxes):
                if i != j:
                    arg_sort = np.argsort(-predicate_preds[i][j])
                    for k in range(num_k):
                        relations.append([i, j])
                        if cfg.TRAIN.USE_SAMPLE_GRAPH: predicates.append(arg_sort[k]+1)
                        else: predicates.append(arg_sort[k])
                        predicate_scores.append(predicate_preds[i][j][arg_sort[k]])

    # predicates = np.argmax(predicate_preds, 2).ravel()
    # predicate_scores = predicate_preds.max(axis=2).ravel()
    # relations = []
    # keep = []
    # if use_gt_rel:
    #     for rel in gt_relations:
    #         i, j = rel[0], rel[1]
    #         keep.append(num_boxes * i + j)
    #         relations.append([i, j])
    # else:
    #     for i in xrange(num_boxes):
    #         for j in xrange(num_boxes):
    #             if i != j:
    #                 keep.append(num_boxes*i + j)
    #                 relations.append([i, j])
    # # take out self relations
    # predicates = predicates[keep]
    # predicate_scores = predicate_scores[keep]

    relations = np.array(relations)
    predicates = np.array(predicates)
    # assert(relations.shape[0] == num_boxes * (num_boxes - 1))
    assert(predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]

    if mode =='pred_cls':
        # if predicate classification task
        # use ground truth bounding boxes
        assert(num_boxes == num_gt_boxes)
        classes = gt_classes
        class_scores = gt_class_scores
        boxes = gt_boxes
    elif mode =='sg_cls':
        assert(num_boxes == num_gt_boxes)
        # if scene graph classification task
        # use gt boxes, but predicted classes

        classes = np.argmax(class_preds, 1)
        class_scores = class_preds.max(axis=1)
        boxes = gt_boxes
    elif mode =='sg_det':
        # if scene graph detection task
        # use preicted boxes and predicted classes
        # classes = np.argmax(class_preds, 1)
        # class_scores = class_preds.max(axis=1)
        # boxes = []
        # for i, c in enumerate(classes):
        #     boxes.append(box_preds[i, c*4:(c+1)*4])
        # boxes = np.vstack(boxes)
        classes = sg_entry['cls_preds']
        class_scores = sg_entry['cls_scores']
        boxes = sg_entry['boxes']
    else:
        raise NotImplementedError('Incorrect Mode! %s' % mode)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(predicates, relations, classes, boxes,
                 predicate_scores, class_scores)


    sorted_inds = np.argsort(relation_scores)[::-1]
    # compue recall
    for k in result_dict[mode + '_recall']:
        # print(k, num_relations)
        this_k = min(k, num_relations)
        keep_inds = sorted_inds[:this_k]
        recall = _relation_recall(gt_triplets,
                                  pred_triplets[keep_inds,:],
                                  gt_triplet_boxes,
                                  pred_triplet_boxes[keep_inds,:],
                                  iou_thresh, num_true_gt_rels=num_true_gt_rels)
        # print(recall)
        result_dict[mode + '_recall'][k].append(recall)

    # for visualization
    return pred_triplets[sorted_inds, :], pred_triplet_boxes[sorted_inds, :]


def _triplet(predicates, relations, classes, boxes,
             predicate_scores, class_scores):

    # format predictions into triplets
    assert(predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]
    triplets = np.zeros([num_relations, 3]).astype(np.int32)
    triplet_boxes = np.zeros([num_relations, 8]).astype(np.int32)
    triplet_scores = np.zeros([num_relations]).astype(np.float32)
    for i in xrange(num_relations):
        triplets[i, 1] = predicates[i]
        sub_i, obj_i = relations[i,:2]
        triplets[i, 0] = classes[sub_i]
        triplets[i, 2] = classes[obj_i]
        triplet_boxes[i, :4] = boxes[sub_i, :]
        triplet_boxes[i, 4:] = boxes[obj_i, :]
        # compute triplet score
        score =  class_scores[sub_i]
        score *= class_scores[obj_i]
        score *= predicate_scores[i]
        triplet_scores[i] = score
    return triplets, triplet_boxes, triplet_scores


def _relation_recall(gt_triplets, pred_triplets,
                     gt_boxes, pred_boxes, iou_thresh, num_true_gt_rels):

    # compute the R@K metric for a set of predicted triplets

    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0

    # ignore duplicated relation
    if num_true_gt_rels is not None:
        num_gt = num_true_gt_rels

    for gt, gt_box in zip(gt_triplets, gt_boxes):
        keep = np.zeros(pred_triplets.shape[0]).astype(bool)
        for i, pred in enumerate(pred_triplets):
            if gt[0] == pred[0] and gt[1] == pred[1] and gt[2] == pred[2]:
                keep[i] = True
        if not np.any(keep):
            continue
        boxes = pred_boxes[keep,:]
        sub_iou = iou(gt_box[:4], boxes[:,:4])
        obj_iou = iou(gt_box[4:], boxes[:,4:])
        inds = np.intersect1d(np.where(sub_iou >= iou_thresh)[0],
                              np.where(obj_iou >= iou_thresh)[0])
        if inds.size > 0:
            num_correct_pred_gt += 1
    return float(num_correct_pred_gt) / float(num_gt)


def iou(gt_box, pred_boxes):
    # computer Intersection-over-Union between two sets of boxes
    ixmin = np.maximum(gt_box[0], pred_boxes[:,0])
    iymin = np.maximum(gt_box[1], pred_boxes[:,1])
    ixmax = np.minimum(gt_box[2], pred_boxes[:,2])
    iymax = np.minimum(gt_box[3], pred_boxes[:,3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) +
            (pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) *
            (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps
