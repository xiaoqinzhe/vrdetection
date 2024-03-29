# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()


# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 256

# Number of negative relationship to sample
__C.TRAIN.NUM_NEG_RELS = 15

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_FREQ = 10000
__C.TRAIN.DISPLAY_FREQ = 10
__C.TRAIN.SUMMARY_FREQ = 50

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''
__C.TRAIN.SNAPSHOT_PREFIX = ''

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = False
__C.TRAIN.BBOX_TARGET_NORMALIZATION_FILE = 'data/vg/bbox_distribution.npy'
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

# Use RPN to detect objects
__C.TRAIN.USE_RPN_DB = False

# Testing options
#

__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.PROPOSAL_NMS = 0.3
__C.TEST.NUM_PROPOSALS = 50

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TEST.FG_THRESH = 0.5

# Propose boxes
__C.TEST.HAS_RPN = False

# Use RPN Database
__C.TEST.USE_RPN_DB = False
#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1./16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3


# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

__C.VRD_DIR = osp.join(__C.ROOT_DIR, 'data/vrd/')
__C.VG_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data/vg/'))

# __C.WORD2VEC_FILE = osp.join(__C.ROOT_DIR, 'data/word2vec/GoogleNews-vectors-negative300.bin')
__C.WORD2VEC_FILE = osp.join(__C.ROOT_DIR, 'data/word2vec/glove.6B.300d.txt')
__C.WORD2VEC_FILE = osp.join(__C.ROOT_DIR, 'data/vrd/')
__C.WORD2VEC_SIZE = 50

# Default GPU device id
__C.GPU_ID = 0

__C.VIZ_DATA_PATH = osp.join(__C.ROOT_DIR, 'data/viz/')

__C.DATASET = 'vg_drnet'
__C.DATASET_DIR = '/hdd/sda/datasets/vrd/'

__C.TRAIN.USE_VALDB = True

# train mode
__C.TRAIN.LEARNING_RATE = 0.001
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.STEPSIZES = [30000, 60000]
# vg
# __C.TRAIN.STEPSIZES = [150000, 300000]

__C.TRAIN.WEIGHT_REG = True
# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.00005

__C.TRAIN.MODE = 'cls'

__C.TRAIN.USE_AUG_DATA = False

__C.TRAIN.USE_SAMPLE_GRAPH = False
__C.TEST.USE_WEIGHTED_REL = False
__C.TEST.USE_PRIOR = True
# __C.TEST.PRIOR_FILENAME = 'lang_prior_graph_1_16.npy'
__C.TEST.PRIOR_FILENAME = 'lang_prior.npy'
# __C.TEST.PRIOR_FILENAME = 'dataset_prior.npy'
__C.TEST.USE_PREDICTION = True
__C.TEST.K_PREDICATE = 1
# zero shot testing
__C.TEST.ZERO_SHOT = False
__C.TEST.USE_GT_REL = True

# sample
__C.TEST.USE_PRIOR = False
__C.TEST.USE_PREDICTION = True
__C.TRAIN.USE_SAMPLE_GRAPH = True
__C.TEST.K_PREDICATE = 24

__C.TRAIN.NUM_NEG_RELS = 128
__C.TRAIN.NUM_SAMPLE_PAIRS = 32

__C.TRAIN.CONV_BP = True

__C.MODEL_PARAMS = {'if_pred_cls': False, 'if_pred_bbox': False, 'if_pred_rel': True, 'if_pred_spt': False,
                    'use_vis': True, 'use_spatial': True, 'use_embedding': True, 'use_class': True,
                    'stop_gradient': True, }

__C.TRAIN.USE_RPN_DB = True
__C.TRAIN.USE_FG_BG = True

# __C.BASENET='res50'
# __C.BASENET_WEIGHT_ITER='75000'
__C.BASENET='vgg16'
__C.BASENET_WEIGHT_ITER='75000'
# __C.BASENET_WEIGHT_ITER='450000'

__C.TEST.REL_EVAL = True
__C.TEST.METRIC_EVAL = True


#
# ResNet options
#

__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize.
# if true, the region will be resized to a square of 2xPOOLING_SIZE,
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1


def get_output_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value