"""
A helper class for evaluating scene graph prediction tasks
"""

import numpy as np
import copy
from sg_eval import eval_relation_recall

class SceneGraphEvaluator:

    def __init__(self, imdb, mode, metrics = None):
        self.imdb = imdb
        self.roidb = imdb.roidb
        self.result_dict = {}
        self.mode = mode

        self.result_dict = {}
        self.result_dict[self.mode + '_recall'] = {20:[], 50:[], 100:[]}
        self.metrics = metrics
        if metrics is not None:
            self.metrics = {}
            for k in metrics: self.metrics[k] = []
        self.rels_to_show = []

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

    def add_rels_to_show(self, rels):
        self.rels_to_show.append(rels)

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

    def save_rels_to_show(self, fn, save_true_pred=True):
        ind2predicate = self.imdb.ind_to_predicates
        ind2class = self.imdb.ind_to_classes
        for i in range(len(self.rels_to_show)):
            rels = self.rels_to_show[i]
            stat = "image id {}, name {}:\n".format(i, self.imdb.info[i]['image_filename'].split('/')[-1])
            for rel in rels:
                stat += "%s -> %s = %s : %s\n" % (ind2class[rel[0]], ind2class[rel[1]], ind2predicate[rel[2]], ind2predicate[rel[3]])
            fn.write(stat)