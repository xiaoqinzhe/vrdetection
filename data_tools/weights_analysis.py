import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

import argparse
from networks.factory import get_network
from vrd_preprocess import get_inds
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', dest='filename',
                        help='network weight filename', type=str,
                        default="./checkpoints/multinet_7/weights_69999.ckpt")
    parser.add_argument('--network', dest='network',
                        help='network name', type=str,
                        default='multinet')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()
    input_pls = {
        'ims': tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3]),
        'rois': tf.placeholder(dtype=tf.float32, shape=[None, 5]),
        'rel_rois': tf.placeholder(dtype=tf.float32, shape=[None, 5]),
        'labels': tf.placeholder(dtype=tf.int32, shape=[None]),
        # 'bboxes': tf.placeholder(dtype=tf.float32, shape=[None, 4]),
        'relations': tf.placeholder(dtype=tf.int32, shape=[None, 2]),
        'predicates': tf.placeholder(dtype=tf.int32, shape=[None]),
        'rel_spts': tf.placeholder(dtype=tf.int32, shape=[None]),
        'bbox_targets': tf.placeholder(dtype=tf.float32, shape=[None, 4 * 101]),
        'bbox_inside_weights': tf.placeholder(dtype=tf.float32, shape=[None, 4 * 101]),
        'num_roi': tf.placeholder(dtype=tf.int32, shape=[]),  # number of rois per batch
        'num_rel': tf.placeholder(dtype=tf.int32, shape=[]),  # number of relationships per batch
        'obj_context_o': tf.placeholder(dtype=tf.int32, shape=[None]),
        'obj_context_p': tf.placeholder(dtype=tf.int32, shape=[None]),
        'obj_context_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
        'rel_context': tf.placeholder(dtype=tf.int32, shape=[None]),
        'rel_context_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
        'obj_embedding': tf.placeholder(dtype=tf.float32, shape=[101, 50]),
        'obj_matrix': tf.placeholder(dtype=tf.float32, shape=[None, None]),
        'rel_matrix': tf.placeholder(dtype=tf.float32, shape=[None, None]),
        'num_classes': 101,
        'num_predicates': 70,
        'num_spatials': 20,
    }
    net = get_network(args.network)(input_pls)
    net.setup()
    restorer = tf.train.Saver()
    with tf.Session() as sess:
        restorer.restore(sess, args.filename)
        vars = tf.global_variables('rel_score')
        print(vars)
        with tf.variable_scope('rel_score', reuse=True):
            weight_v = tf.get_variable("weight")
            # bias_v = tf.get_variable('biases')
        weight = sess.run([weight_v])[0]
        print(weight.shape)

    predicate_to_ind, ind_to_predicate = get_inds('/hdd/datasets/vrd/vrd/predicates.json')
    weight = weight.transpose()

    #
    tsne = TSNE(n_components=2)
    norm_w = tsne.fit_transform(weight)
    plt.scatter(norm_w[:, 0], norm_w[:, 1])
    for i in range(len(norm_w)):
        plt.annotate(ind_to_predicate[i], norm_w[i], fontsize=8)
    plt.savefig('./data_tools/fig/pred_weights_show.png')