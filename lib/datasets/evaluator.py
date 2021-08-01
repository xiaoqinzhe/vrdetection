"""
A helper class for evaluating scene graph prediction tasks
"""

import numpy as np
import copy
from datasets.sg_eval import eval_relation_recall
from datasets.viz_eval import viz_relation
from datasets.eval_utils import EvalRunnerMP

class SceneGraphEvaluator:

    def __init__(self, imdb, mode, metrics = None, prior=None, top_k=1, use_prediction=True, use_prior=False, use_weight=False, use_fg_weight=False):
        self.imdb = imdb
        self.roidb = imdb.roidb
        self.result_dict = {}
        self.mode = mode
        self.prior = prior

        self.result_dict = {}
        self.result_dict[self.mode + '_recall'] = {10:[], 20:[], 50:[], 100:[]}
        self.metrics = metrics
        if metrics is not None:
            self.metrics = {}
            for k in metrics: self.metrics[k] = []
        self.top_k=top_k
        self.use_prediction=use_prediction
        self.use_prior=use_prior
        self.use_weight=use_weight
        self.use_fg_weight = use_fg_weight
        self._weight = None
        self.rels_to_show = []
        self.num_samples = len(self.roidb)

        self.runner = EvalRunnerMP(self.mp_eval_once,
                                   task_capacity=10, data_capacity=10)
        self.runner.start_processes(1)

    def mp_eval_once(self, data):
        result_dict = {}
        result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
        eval_relation_recall(data[0], self.roidb[data[1]],
                             result_dict,
                             self.mode,
                             iou_thresh=0.5,
                             top_k=self.top_k, use_prediction=self.use_prediction,
                             use_prior=self.use_prior, use_weight=self.use_weight, use_fg_weight=self.use_fg_weight,
                             prior=self.prior
                             )
        return result_dict

    def evaluate_scene_graph_entry(self, sg_entry, im_idx, iou_thresh=0.5, viz=False):
        sg_entry["num_predicates"] = self.imdb.num_predicates
        sg_entry["num_classes"] = self.imdb.num_classes
        self.runner.put_task([sg_entry, im_idx])
        results = self.runner.get_data()
        # print(results)
        for res in results:
            for k1 in self.result_dict:
                for k2 in self.result_dict[k1]:
                    self.result_dict[k1][k2].append(res[k1][k2])
        # if viz:
        #     viz_relation(sg_entry, self.imdb, im_idx,
        #                          self.result_dict, prior=prior )
        # else:
        #     pred_triplets, triplet_boxes = \
        #     eval_relation_recall(sg_entry, self.roidb[im_idx],
        #                         self.result_dict,
        #                         self.mode,
        #                         iou_thresh=iou_thresh,
        #                         top_k=self.top_k, use_prediction=self.use_prediction,
        #                         use_prior=self.use_prior, use_weight=self.use_weight, use_fg_weight=self.use_fg_weight, prior=prior
        #                          )
        #     return pred_triplets, triplet_boxes

    def add_metrics(self, metric):
        if self.metrics is None: return
        for k in metric: self.metrics[k].append(metric[k])

    def add_rels_to_show(self, rels):
        self.rels_to_show.append(rels)

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        cur_size = len(self.result_dict[self.mode+'_recall'][50])
        while cur_size < self.num_samples:
            results = self.runner.get_data()
            for res in results:
                for k1 in self.result_dict:
                    for k2 in self.result_dict[k1]:
                        self.result_dict[k1][k2].append(res[k1][k2])
            cur_size += len(results)
        print("num_samples: {}".format(len(self.result_dict[self.mode+'_recall'][50])))
        print('====== ' + self.mode
              + " use_prediction " + str(self.use_prediction)
              + " top_k " + str(self.top_k)
              + " use_prior " + str(self.use_prior)
              + " use_weight " + str(self.use_weight)
              + " use_fg_weight " + str(self.use_fg_weight)
              + ' =========')
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