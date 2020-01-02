#!/usr/bin/env python

# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# Adapted from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

import _init_paths
from fast_rcnn.train import train_net
from fast_rcnn import train_tl
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_db, get_val_db
from roi_data_layer.roidb import prepare_roidb, compute_bbox_target_normalization
import argparse
import pprint
import numpy as np
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a scene graph generation network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        default=None, type=str)
    parser.add_argument('--output', dest='output_dir',
                        default='output/test')
    parser.add_argument('--tf_log', dest='tf_log',
                        default='output/tf_logs/test')
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default="vggnet", type=str)
    parser.add_argument('--inference_iter', dest='inference_iter',
                        default=2, type=int)
    parser.add_argument('--train_type', dest='train_type',
                        default="normal", type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.TRAIN.INFERENCE_ITER = args.inference_iter

    if args.dataset is not None:
        cfg['DATASET'] = args.dataset

    if args.network_name in ["weightnet", "ranknet", 'ctxnet', 'graphnet', 'attnet']:
        cfg.TRAIN.USE_GRAPH_SAMPLE=True
    else: cfg.TRAIN.USE_GRAPH_SAMPLE=False

    if 'vg' in args.dataset:
        cfg.TRAIN.STEPSIZES = [150000, 300000]
        cfg.BASENET_WEIGHT_ITER = '450000'

    if args.dataset == 'tl_vg':
        cfg.TRAIN.STEPSIZES = [50000, 100000]
        cfg.BASENET_WEIGHT_ITER = '150000'

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    imdb = get_db()
    print('Loaded db `{:s}` for training'.format(cfg.DATASET))
    if cfg.TRAIN.USE_FLIPPED:
        print('appending flipped images')
        imdb.append_flipped_images()
    roidb = imdb.roidb
    print('roidb loaded')

    if cfg.TRAIN.USE_VALDB:
        val_db = get_val_db()
        val_roidb = val_db.roidb
    else: val_roidb = None

    # compute bbox target mean and stds if not precomputed
    if False:
        print('precomputing target means...')
        #print(len(roidb), roidb[0].keys(), roidb[0]['boxes'].shape)
        imdb.add_rpn_rois(roidb, make_copy=False)
        #print(len(roidb), roidb[0].keys(), roidb[0]['boxes'].shape)
        prepare_roidb(roidb)
        bbox_means, bbox_stds = compute_bbox_target_normalization(roidb)
        print(bbox_means)
        print(bbox_stds)
        np.save(cfg.TRAIN.BBOX_TARGET_NORMALIZATION_FILE, {'means': bbox_means, 'stds': bbox_stds})

    device_name = '/gpu:{:d}'.format(args.gpu_id)
    print(device_name)
    if args.train_type == "tl":
        train_tl.train_net(args.network_name, imdb, roidb, args.output_dir, args.tf_log,
                  pretrained_model=args.pretrained_model,
                  max_iters=args.max_iters, val_roidb=val_roidb)
    else:
        train_net(args.network_name, imdb, roidb, args.output_dir, args.tf_log,
                  pretrained_model=args.pretrained_model,
                  max_iters=args.max_iters, val_roidb=val_roidb)

