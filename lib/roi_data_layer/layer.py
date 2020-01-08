from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
from roi_data_layer.roidb import prepare_roidb, add_bbox_regression_targets
import numpy as np

class RoIDataLayer:
    def __init__(self, imdb, roidb, num_classes, bbox_means, bbox_stds, num_batches=cfg.TRAIN.IMS_PER_BATCH):
        self.imdb = imdb
        self._roidb = roidb
        self._num_classes = num_classes
        self._num_batches = num_batches
        self.bbox_means = bbox_means
        self.bbox_stds = bbox_stds
        self._shuffle_roidb_inds()
        # self.wordvec = load_word_vec(imdb.ind_to_predicates)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            if inds.shape[0] % self._num_batches != 0:
                num = self._num_batches - (inds.shape[0] % self._num_batches)
                inds = np.hstack((inds, np.random.choice(vert_inds, size=num)))
            inds = np.reshape(inds, (-1, self._num_batches))     ## 2????????????
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self._num_batches >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self._num_batches]
        self._cur += self._num_batches
        return db_inds

    def _get_next_minibatch(self, db_inds):
        """Return the blobs to be used for the next minibatch.
        """
        minibatch_db = [self._roidb[i] for i in db_inds]
        if cfg.TRAIN.USE_RPN_DB:
            minibatch_db = self.imdb.add_rpn_rois(minibatch_db)
        prepare_roidb(minibatch_db)
        add_bbox_regression_targets(minibatch_db, self.bbox_means,
                                    self.bbox_stds)

        blobs = get_minibatch(minibatch_db, self._num_classes)
        # if blobs is not None:
        #     blobs['db_inds'] = db_inds
        return blobs

    def next_batch(self):
        """Get blobs and copy them into this layer's top blob vector."""
        batch = None
        while batch is None:
            db_inds = self._get_next_minibatch_inds()
            batch = self._get_next_minibatch(db_inds)
        return batch
