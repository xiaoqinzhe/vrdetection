from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from fast_rcnn.config import cfg

class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._roidb = None
        self.ind_to_classes = []
        self.ind_to_predicates = []
        self._prior = None

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self.ind_to_classes)

    @property
    def classes(self):
        return self.ind_to_classes

    @property
    def predicates(self):
        return self.ind_to_predicates

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    @property
    def roidb(self):
        if self._roidb is not None:
            return self._roidb
        self._roidb = self._roidb_handler()
        return self._roidb

    @property
    def num_images(self):
      return len(self.image_index)

    def _get_widths(self):
        raise NotImplementedError

    @property
    def prior(self):
        if self._prior is None:
            path = './data/' + cfg.DATASET
            if cfg.DATASET.startswith('vg'):
                path = './data/vg/' + cfg.DATASET
            filename = path + "/" + cfg.TEST.PRIOR_FILENAME
            self._prior = np.load(filename)
            # print("load prior from {}".format(filename))
        return self._prior

    def get_spatial_info(self, file):
        rep = {}
        rep2 = {0:'ignore'}
        with open(file) as f:
            for i in range(7):
                line = f.readline()
                preds = line.split(',')
                for j in range(len(preds)):
                    preds[j] = preds[j].lstrip(' ').rstrip(' ').rstrip('\n')
                    rep[preds[j]] = i + 1
                rep2[i + 1] = preds[0]
        return rep, rep2

    def get_spatial_class(self, rels, ind_to_predicates, replace_list):
        sp = []
        for rel in rels:
            predicate_name = ind_to_predicates[rel[2]]
            if predicate_name in replace_list:
                sp.append(replace_list[predicate_name])
            else: sp.append(0)
        return np.array(sp)

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == len(gt_roidb), \
                'Number of boxes must match number of ground-truth roidb'
        roidb = []
        for i, boxes in enumerate(box_list):
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                            gt_boxes.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({
                'boxes' : boxes,
                'gt_overlaps' : overlaps,
                'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
                'flipped' : False,
                'seg_areas' : np.zeros((num_boxes,), dtype=np.float32)
            })
        return roidb

    @staticmethod
    def merge_gt_rpn_roidb(gt_roidb, rpn_roidb):
        assert len(gt_roidb) == len(rpn_roidb)
        for i in range(len(gt_roidb)):
            gt_roidb[i]['boxes'] = np.vstack((gt_roidb[i]['boxes'], rpn_roidb[i]['boxes']))
            gt_roidb[i]['gt_overlaps'] = scipy.sparse.vstack([gt_roidb[i]['gt_overlaps'],
                                                       rpn_roidb[i]['gt_overlaps']])
            gt_roidb[i]['gt_classes'] = np.hstack((gt_roidb[i]['gt_classes'],rpn_roidb[i]['gt_classes']))
            gt_roidb[i]['seg_areas'] = np.hstack((gt_roidb[i]['seg_areas'],
                                           rpn_roidb[i]['seg_areas']))
            gt_roidb[i]['roi_scores'] = np.hstack([gt_roidb[i]['roi_scores'], rpn_roidb[i]['roi_scores']])
        return gt_roidb

    def append_flipped_images(self):
        num_images = self.num_images
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = self.roidb[i]['width'] - oldx2 - 1
            boxes[:, 2] = self.roidb[i]['width'] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {}
            for key in self.roidb[i]:
                entry[key] = self.roidb[i][key]
            entry['boxes'] = boxes
            entry['flipped'] = True

            self.roidb.append(entry)
        self._image_index = np.hstack([self._image_index,
                                       self._image_index]).transpose()
        # self.im_sizes = np.vstack([self.im_sizes,
        #                            self.im_sizes])
