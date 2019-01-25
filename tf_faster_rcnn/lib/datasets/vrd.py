# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg, get_output_dir
import json, pickle
from utils.cython_bbox import bbox_overlaps
from datasets.eval import draw_det_images, draw_gt_images, mAP, maxRelPrecision

class vrd(imdb):
  def __init__(self, image_set):
    name = 'vrd_' + image_set
    imdb.__init__(self, name)
    self._image_set = image_set
    self._data_path = '../data/vrd'
    # image_set='train'
    if image_set == 'train':
        json_file = self._data_path + "/train.json"
        # json_file = self._data_path + "/train_aug.pickle"
    else:
        json_file = self._data_path + "/test.json"
    if json_file.endswith("json"):
        self.info = json.load(open(json_file))
    else: self.info = pickle.load(open(json_file))
    # projection between class/predicate and idx, add background to class
    self._class_to_ind = {"background": 0}
    self._classes = ["background"]
    for i, name in enumerate(self.info['ind_to_class']):
        self._class_to_ind[name] = i + 1
        self._classes.append(name)
    self.info = self.info['data']
    self._image_index = np.arange(len(self.info))
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = "/hdd/datasets/vrd/vrd/"+self.info[index]['image_filename']
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    gt_roidb = []
    for i in range(self.num_images):
        boxes = np.array(self.info[i]['boxes'], dtype=np.float32)
        if boxes.shape[0] == 0:
            continue
        gt_classes = np.array(self.info[i]['labels']) + 1
        overlaps = np.zeros((len(boxes), self.num_classes))
        for j, o in enumerate(overlaps):  # to one-hot
            o[gt_classes[j]] = 1.0
        overlaps = scipy.sparse.csr_matrix(overlaps)
        relation = np.array(self.info[i]["relations"])

        seg_areas = np.multiply((boxes[:, 2] - boxes[:, 0] + 1),
                                (boxes[:, 3] - boxes[:, 1] + 1))  # box areas
        gt_roidb.append({'boxes': boxes,
                         'gt_classes': gt_classes,
                         'gt_overlaps': overlaps,
                         'flipped': False,
                         'seg_areas': seg_areas,
                         'width': self.info[i]['image_width'],
                         'height': self.info[i]['image_height']})
    return gt_roidb

  def gt_rels(self):
      gt_rels = []
      for i in range(self.num_images):
          relation = np.array(self.info[i]["relations"])
          gt_rels.append(relation)
      return gt_rels

  def _get_widths(self):
    return [self.info[i]['image_width']
            for i in range(self.num_images)]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}
      self.roidb.append(entry)
    self._image_index = np.hstack([self._image_index, self._image_index])

  def evaluate_detections(self, all_boxes, output_dir=None):
      draw_gt_images(self, get_output_dir(self, "det_images"), num_images=-1)
      print("implementation is coming soon.")
      iou_threshhold = np.arange(0.0, 1.0, 0.1)
      cls_score_threshold = 0.0
      gt_roidb = self.roidb
      gt_count = np.zeros([len(gt_roidb), len(iou_threshhold)], dtype=np.float32)
      for im_i in range(len(gt_roidb)):

          # l = 0
          # for j in range(len(all_boxes)):
          #     l += len(all_boxes[j][im_i])
          # print(l)

          gt_boxes = gt_roidb[im_i]['boxes']
          gt_cls = gt_roidb[im_i]['gt_classes']
          for i in range(len(gt_boxes)):
              # print(im_i, i)
              cls = gt_cls[i]
              b = np.array(all_boxes[cls][im_i])
              if len(b) == 0: continue
              score_inds = np.where(b[:, 4]>cls_score_threshold)[0]
              if len(score_inds) == 0: continue
              b = b[score_inds]
              pred_boxes = b[:, :4]
              overlaps = bbox_overlaps(gt_boxes[i:i+1].astype(np.float), pred_boxes.astype(np.float))[0]
              # print(overlaps)
              for j, iou_t in enumerate(iou_threshhold):
                  inds = np.where(overlaps>iou_t)[0]
                  if len(inds) > 0:
                      gt_count[im_i, j] += 1.0
          # print(gt_count[im_i])
          gt_count[im_i] = gt_count[im_i] / len(gt_boxes)
      avg = np.average(gt_count, axis=0)
      stat = ""
      for i in range(len(iou_threshhold)):
          stat += "{}: {} ".format(iou_threshhold[i], avg[i])
      print(stat)
      map = mAP(self.roidb, all_boxes, cls_score_threshold=cls_score_threshold)
      print("map: ", map)
      maxRelPrecision(self.roidb, self.gt_rels(), all_boxes, cls_score_threshold=cls_score_threshold)
      # draw_det_images(self, all_boxes, get_output_dir(self, "det_images"), num_images=-1)

if __name__ == '__main__':
  from datasets.pascal_voc import pascal_voc

  d = pascal_voc('trainval', '2007')
  res = d.roidb
  from IPython import embed;

  embed()
