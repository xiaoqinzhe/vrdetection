"""
A helper class for evaluating scene graph prediction tasks
"""

import numpy as np
import copy
from sg_eval import eval_relation_recall

class SceneGraphEvaluator:

    def __init__(self, imdb, mode, metrics = None):
        self.roidb = imdb.roidb
        self.result_dict = {}
        self.mode = mode

        self.result_dict = {}
        self.result_dict[self.mode + '_recall'] = {20:[], 50:[], 100:[]}
        self.metrics = metrics
        if metrics is not None:
            self.metrics = {}
            for k in metrics: self.metrics[k] = []

    def evaluate_scene_graph_entry(self, sg_entry, im_idx, iou_thresh):
        pred_triplets, triplet_boxes = \
            eval_relation_recall(sg_entry, self.roidb[im_idx],
                                self.result_dict,
                                self.mode,
                                iou_thresh=iou_thresh)
        return pred_triplets, triplet_boxes

    def add_metrics(self, metric):
        assert self.metrics is not None
        for k in metric: self.metrics[k].append(metric[k])

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))
        if self.metrics is not None:
            stat = ''
            for k in self.metrics:
                stat += '-%s-: %f ' % (k, np.mean(self.metrics[k]))
            print('metrics: %s' % stat)
