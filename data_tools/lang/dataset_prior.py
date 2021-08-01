import numpy as np
import json
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
from datasets.vg import vg
from fast_rcnn.config import cfg
import scipy.sparse
from sklearn.preprocessing import normalize


def get_oo_id(i, j, num_class):
    return i * num_class + j


def reverse_id(id, num_class):
    return id // num_class, id % num_class


def get_vrd_o2pm(info):
    num_class = len(info['ind_to_class'])
    num_predicate = len(info['ind_to_predicate'])

    o2p_m = np.zeros([num_class, num_class, num_predicate], np.float32)
    data = info['data']
    for im in range(len(data)):
        label = data[im]['labels']
        for i in range(len(data[im]['relations'])):
            rel = data[im]['relations'][i]
            o2p_m[label[rel[0]], label[rel[1]], rel[2]] += 1
    k = 0.1
    o2p_m = (o2p_m+k)/(np.sum(o2p_m, axis=-1, keepdims=True)+k*num_predicate)
    return o2p_m

def get_vrd_o2om(info):
    num_class = len(info['ind_to_class'])
    o2o_m = np.zeros([num_class, num_class])
    data = info['data']
    for im in range(len(data)):
        label = data[im]['labels']
        for i in range(len(data[im]['relations'])):
            rel = data[im]['relations'][i]
            o2o_m[label[rel[0]], label[rel[1]]] += 1.0
    k = 0.1
    o2o_m = (o2o_m+k) / (np.sum(o2o_m, -1, keepdims=True)+k*num_class)
    return o2o_m

def get_vrd_om(info):
    num_class = len(info['ind_to_class'])
    o_m = np.zeros([num_class])
    data = info['data']
    for im in range(len(data)):
        label = data[im]['labels']
        for i in range(len(data[im]['relations'])):
            rel = data[im]['relations'][i]
            o_m[label[rel[0]]] += 1.0
            o_m[label[rel[1]]] += 1.0
    o_m /= np.sum(o_m)
    return o_m

def get_matrix(ind2class, o2p_m, w2v_filename):
    num_class = len(ind2class)
    # div = np.reshape(np.tile(np.sum(o2p_m, axis=1) + 1, [o2p_m.shape[1]]),
    #                 [o2p_m.shape[1], o2p_m.shape[0]]).transpose(i)
    div = np.sum(o2p_m, axis=1)[:, np.newaxis] + 0.00000001
    o2p_m = o2p_m / div
    w2v = np.load(w2v_filename)
    class_v = w2v[:len(ind2class), :]
    oo_v = np.zeros([num_class ** 2, w2v.shape[1] * 2])
    for i in range(num_class):
        for j in range(num_class):
            oo_v[get_oo_id(i, j, num_class)] = np.hstack((class_v[i], class_v[j]))
    oo_v = normalize(oo_v)
    o2o_m = np.matmul(oo_v, oo_v.transpose())
    o2o_m /= np.sum(o2o_m, axis=1)[:, np.newaxis]
    # o2o_m = np.exp(np.matmul(oo_v, oo_v.transpose()))
    # o2o_m_t = (np.sum(o2o_m, axis=1)-np.diag(o2o_m))[:,np.newaxis]
    # o2o_m -= np.diag(np.diag(o2o_m))
    # o2o_m /= o2o_m_t

    # new_o2p_m = np.matmul(o2o_m, o2p_m)

    # new_o2p_m2 = np.matmul(o2o_m, new_o2p_m)
    # new_o2p_m3 = np.matmul(o2o_m, new_o2p_m2)
    # new_o2p_m = new_o2p_m + new_o2p_m2 + new_o2p_m3

    lanta = 0.5
    new_o2p_m = (1 - lanta) * np.matmul(np.linalg.inv(np.eye(o2o_m.shape[0], o2o_m.shape[1]) - lanta * o2o_m), o2p_m)

    # new_o2p_m = new_o2p_m + o2p_m
    # new_o2p_m = o2p_m

    # rows norm
    new_o2p_m = new_o2p_m / np.sum(new_o2p_m, axis=1)[:, np.newaxis]
    # print(new_o2p_m[55])
    return o2p_m, new_o2p_m


def save(lang_o2p_m, filename, o2p_m=None, d_filename=None):
    np.save(filename, lang_o2p_m)
    if o2p_m is not None:
        np.save(d_filename, o2p_m)
    print("saved successful to {}!".format(filename))


def show_o2p(origin_o2p, o2p, ind2class, ind2predicate, num_show=1000):
    for i in range(len(ind2predicate)):
        while (True):
            # ro = np.random.randint(0, len(origin_o2p))
            # rp = np.random.randint(0, len(origin_o2p[0]))

            # if o2p[ro, rp] > 0.1: break;
            ro, rp = 0 * len(ind2class) + 41, i
            break
        l1, l2 = reverse_id(ro, len(ind2class))
        if o2p[ro, rp] > 0.01:
            print(ind2class[l1], ind2class[l2], ind2predicate[rp], origin_o2p[ro, rp], o2p[ro, rp])


def vrd_prior(data_dir, suffix=''):
    info = json.load(open('./data/{}/train.json'.format(data_dir)))
    o2p_m = get_vrd_o2pm(info)
    o2o_m = get_vrd_o2om(info)
    o_m = get_vrd_om(info)
    print(np.sum(o2p_m[0,1]), np.sum(o2o_m[0]), np.sum(o_m))
    np.save("./data/{}/dataset_gp".format(data_dir), {"o2p_m":o2p_m, "o2o_m":o2o_m, "o_m":o_m})

if __name__ == '__main__':
    vrd_prior('vg/vg_vtranse', '')
