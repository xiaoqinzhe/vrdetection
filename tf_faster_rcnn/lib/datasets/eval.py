from utils.cython_bbox import bbox_overlaps
from utils.visualization import _draw_single_box, FONT, STANDARD_COLORS, NUM_COLORS
from six.moves import range
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np
import cv2, os

def draw_det_images(imdb, all_boxes, save_path, num_images=-1):
    ind2class = imdb.classes
    if num_images == -1: num_images = imdb.num_images
    for im_i in range(num_images):
        boxes, clses = np.zeros([0, 4], np.float), np.zeros([0], np.float)
        scores = np.zeros([0], np.float)
        for j in range(imdb.num_classes):
            if len(all_boxes[j][im_i]) == 0: continue
            boxes = np.vstack([boxes, np.array(all_boxes[j][im_i])[:,:4]])
            clses = np.hstack([clses, np.zeros([len(all_boxes[j][im_i])])+j])
            scores = np.hstack([scores, np.array(all_boxes[j][im_i])[:,4]])
        boxes = np.hstack((boxes, clses[:, np.newaxis]))
        image = cv2.imread(imdb.image_path_at(im_i))
        image = draw_image(image, boxes, ind2class, scores)
        cv2.imwrite(save_path+"/{}_det.jpg".format(im_i), image)
    print("save images detected boxes into {} done.".format(save_path))

def draw_gt_images(imdb, save_path, num_images=-1):
    ind2class = imdb.classes
    if num_images == -1: num_images = imdb.num_images
    for im_i in range(num_images):
        boxes = np.hstack((imdb.roidb[im_i]['boxes'], imdb.roidb[im_i]['gt_classes'][:, np.newaxis]))
        image = cv2.imread(imdb.image_path_at(im_i))
        image = draw_image(image, boxes, ind2class)
        basename=os.path.basename(imdb.info[im_i]['image_filename'])
        # if '337085136_ac59abf2d6_b' in basename or '1589782798_1412fcabd5_o' in basename or '321785477_a4a4739ee5_o' in basename:
        cv2.imwrite(save_path+"/{}".format(basename), image)
        # print(save_path+"/{}_gt.jpg".format(imdb.info[im_i]['image_filename']))
    print("save gt images into {} done.".format(save_path))

def draw_image(image, boxes, ind2class, scores=None):
    num_boxes = boxes.shape[0]
    gt_boxes_new = boxes.copy()
    gt_boxes_new[:, :4] = np.round(gt_boxes_new[:, :4].copy())
    disp_image = Image.fromarray(np.uint8(image))
    # print(np.shape(image), np.shape(disp_image))
    for i in range(num_boxes):
        this_class = int(gt_boxes_new[i, 4])
        info = 'N%02d-%s' % (i, ind2class[this_class]) if scores is None else 'N%02d-%s, %f' % (i, ind2class[this_class], scores[i])
        disp_image = _draw_single_box(disp_image,
                                      gt_boxes_new[i, 0],
                                      gt_boxes_new[i, 1],
                                      gt_boxes_new[i, 2],
                                      gt_boxes_new[i, 3],
                                      info,
                                      FONT,
                                      color=STANDARD_COLORS[this_class % NUM_COLORS])

    image = np.array(disp_image)
    return image

def maxRelPrecision(gt_roidb, gt_rels, all_boxes, iou_threshhold=0.5, cls_score_threshold=0.5):
    precisions = []
    for im_i in range(len(gt_roidb)):
        gt_boxes = gt_roidb[im_i]['boxes']
        gt_cls = gt_roidb[im_i]['gt_classes']
        count = [0 for _ in range(len(gt_boxes))]
        pred_boxes = [[] for _ in range(len(gt_boxes))]
        print([all_boxes[c][0] for c in range(len(all_boxes))])
        exit()
        for i in range(len(gt_boxes)):
            # print(im_i, i)
            cls = gt_cls[i]
            b = np.array(all_boxes[cls][im_i])
            if len(b) == 0: continue
            score_inds = np.where(b[:, 4] > cls_score_threshold)[0]
            if len(score_inds) == 0: continue
            b = b[score_inds]
            pb = b[:, :4]
            overlaps = bbox_overlaps(gt_boxes[i:i + 1].astype(np.float), pb.astype(np.float))[0]
            # print(overlaps)
            inds = np.where(overlaps > iou_threshhold)[0]
            if len(inds) > 0:
                count[i] = len(inds)
                pred_boxes[i] = pb[inds]
        prec = 0.
        for gt_rel in gt_rels[im_i]:
            if count[gt_rel[0]] > 0 and count[gt_rel[1]] > 0:
                prec += 1
        prec /= len(gt_rels[im_i])
        precisions.append(prec)
        if prec == 0: print(im_i)
    prec = np.mean(precisions)
    print("max rel precision", prec)
    return prec

def mAP(gt_roidb, all_boxes, iou_threshhold=0.5, cls_score_threshold=0.0):
    num_classes = len(all_boxes)
    gt_count = [[] for _ in range(num_classes)]
    for im_i in range(len(gt_roidb)):
        gt_boxes = gt_roidb[im_i]['boxes']
        gt_cls = gt_roidb[im_i]['gt_classes']
        count = [[] for _ in range(num_classes)]
        for i in range(len(gt_boxes)):
            # print(im_i, i)
            cls = gt_cls[i]
            count[cls].append(0.)
            b = np.array(all_boxes[cls][im_i])
            if len(b) == 0: continue
            score_inds = np.where(b[:, 4] > cls_score_threshold)[0]
            if len(score_inds) == 0: continue
            b = b[score_inds]
            pred_boxes = b[:, :4]
            overlaps = bbox_overlaps(gt_boxes[i:i + 1].astype(np.float), pred_boxes.astype(np.float))[0]
            # print(overlaps)
            inds = np.where(overlaps > iou_threshhold)[0]
            if len(inds) > 0:
                count[cls][-1] = 1.0
        for i in range(num_classes):
            if len(count[i]) == 0: continue
            gt_count[i].append(sum(count[i]) / len(count[i]))
    precision = []
    for i in range(1, num_classes):
        if len(gt_count[i]) == 0:
            print("class {} == 0!!!!".format(i))
            continue
        precision.append(sum(gt_count[i]) / len(gt_count[i]))
    return sum(precision) / (len(precision))