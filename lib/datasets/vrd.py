from imdb import imdb
import numpy as np
from fast_rcnn.config import cfg
import os, json, h5py, cv2, scipy.sparse, copy, pickle

class vrd(imdb):
    def __init__(self, data_path, split=0, num_im=-1):
        super(vrd, self).__init__("vrd_dataset")

        if split == 0:
            if not cfg.TRAIN.USE_AUG_DATA:
                json_file = data_path + "/train.json"
            else: json_file = data_path + "/train_aug.pickle"
        else:
            json_file = data_path + "/test.json"
        if json_file.endswith("json"):
            self.info = json.load(open(json_file))
        else:
            self.info = pickle.load(open(json_file))
        # projection between class/predicate and idx, add background to class
        if False:  # ????
            self.class_to_ind = {"background":0}
            self.ind_to_classes = ["background"]
            for i, name in enumerate(self.info['ind_to_class']):
                self.class_to_ind[name] = i + 1
                self.ind_to_classes.append(name)
        else:
            self.class_to_ind = self.info['class_to_ind']
            self.ind_to_classes = self.info['ind_to_class']
        if cfg.TRAIN.USE_SAMPLE_GRAPH:
            self.predicate_to_ind = {"background": 0}
            self.ind_to_predicates = ["background"]
            for i, name in enumerate(self.info['ind_to_predicate']):
                self.predicate_to_ind[name] = i + 1
                self.ind_to_predicates.append(name)
            # print(self.info['data'][0]['relations'])
            for roi in self.info['data']:
                for i in range(len(roi['relations'])): roi['relations'][i][2] += 1
            # print(self.info['data'][0]['relations'])
        else:
            self.predicate_to_ind = self.info['predicate_to_ind']
            self.ind_to_predicates = self.info['ind_to_predicate']

        # self.word2vec = np.load(data_path + '/w2v_all_graph_16.npy')
        self.word2vec = np.load(data_path + '/w2v_all.npy')
        self.embedding_size = self.word2vec.shape[1]
        print("load word2vec from "+data_path + '/w2v.npy')
        # if cfg.TRAIN.USE_SAMPLE_GRAPH:
        # vecs = np.zeros([self.word2vec.shape[0]+1, self.word2vec.shape[1]], np.float32)
        # vecs[1:, :] = self.word2vec
        # self.word2vec = vecs

        self.spatial_to_ind, self.ind_to_spatials = self.get_spatial_info(data_path + '/spatial_alias.txt')
        self.num_spatials = len(self.ind_to_spatials)

        self.info = self.info['data']

        self._image_index = np.arange(0, len(self.info), 1)
        if num_im > -1:
            self._image_index = self._image_index[:num_im]

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

    def im_getter(self, idx):
        im = cv2.imread(self.info[idx]['image_filename'])
        return im

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        """
        gt_roidb = []
        for i in range(self.num_images):
            boxes = np.array(self.info[i]['boxes'], dtype=np.float32)
            if boxes.shape[0] == 0:
                continue
            gt_classes = np.array(self.info[i]['labels'])
            overlaps = np.zeros((len(boxes), self.num_classes))
            for j, o in enumerate(overlaps): # to one-hot
                o[gt_classes[j]] = 1.0
            overlaps = scipy.sparse.csr_matrix(overlaps)
            relation = np.array(self.info[i]["relations"], dtype=np.int32)

            seg_areas = np.multiply((boxes[:, 2] - boxes[:, 0] + 1),
                                    (boxes[:, 3] - boxes[:, 1] + 1)) # box areas
            gt_roidb.append({'boxes': boxes,
                             'gt_classes' : gt_classes,
                             'gt_overlaps' : overlaps,
                             'gt_relations': relation,
                             'gt_spatial': self.get_spatial_class(relation, self.ind_to_predicates, self.spatial_to_ind),
                             'flipped' : False,
                             'seg_areas' : seg_areas,
                             'db_idx': i,
                             'image': lambda im_i=i: self.im_getter(im_i),
                             'roi_scores': np.ones(boxes.shape[0]),
                             'width': self.info[i]['image_width'],
                             'height': self.info[i]['image_height']})
        return gt_roidb

    def add_rpn_rois(self, gt_roidb_batch, make_copy=True):
        """
        Load precomputed RPN proposals
        """
        gt_roidb = copy.deepcopy(gt_roidb_batch) if make_copy else gt_roidb_batch
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_gt_rpn_roidb(gt_roidb, rpn_roidb)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        # load an precomputed ROIDB to the current gt ROIDB
        box_list = []
        score_list = []
        for entry in gt_roidb:
            i = entry['db_idx']
            im_rois = self.rpn_rois[i].copy()
            roi_scores = self.rpn_scores[i].copy()
            box_list.append(im_rois)
            score_list.append(roi_scores)
        roidb = self.create_roidb_from_box_list(box_list, gt_roidb)
        for i, rdb in enumerate(roidb):
            rdb['roi_scores'] = score_list[i]
        return roidb