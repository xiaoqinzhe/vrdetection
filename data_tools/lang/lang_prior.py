import numpy as np
import json
import os, sys
sys.path.append(os.path.join( os.path.dirname(__file__), '..', '..', 'lib' ))
from datasets.vg import vg
from fast_rcnn.config import cfg
import scipy.sparse
from sklearn.preprocessing import normalize

def get_oo_id(i, j, num_class):
    return i*num_class + j

def reverse_id(id, num_class):
    return id//num_class, id%num_class

def get_vrd_o2pm(info):
    num_class = len(info['ind_to_class'])
    num_predicate = len(info['ind_to_predicate'])
    
    o2p_m = np.zeros([num_class ** 2, num_predicate], np.float32)
    data = info['data']
    for im in range(len(data)):
        label = data[im]['labels']
        for i in range(len(data[im]['relations'])):
            rel = data[im]['relations'][i]
            o2p_m[get_oo_id(label[rel[0]], label[rel[1]], num_class), rel[2]] += 1
    # print(sys.getsizeof(o2p_m))
    # o2p_m = scipy.sparse.csr_matrix(o2p_m)
    # print(sys.getsizeof(o2p_m))
    return o2p_m

def get_vg_info():
    cfg.TRAIN.USE_SAMPLE_GRAPH = False
    vgd = vg('./data/vg/', split=0)
    num_class = len(vgd.ind_to_classes)
    num_predicate = len(vgd.ind_to_predicates)
    roidb = vgd.roidb
    o2p_m = np.zeros([num_class ** 2, num_predicate])
    for im in range(len(roidb)):
        label = roidb[im]['gt_classes']
        for i in range(len(roidb[im]['gt_relations'])):
            rel = roidb[im]['gt_relations'][i]
            o2p_m[get_oo_id(label[rel[0]], label[rel[1]], num_class), rel[2]] += 1.0
    return o2p_m, vgd.ind_to_classes, vgd.ind_to_predicates

def get_vrd_o2om(info):
    num_class = len(info['ind_to_class'])
    o2o_m = np.zeros([num_class, num_class])
    data = info['data']
    for im in range(len(data)):
        label = data[im]['labels']
        for i in range(len(data[im]['relations'])):
            rel = data[im]['relations'][i]
            o2o_m[label[rel[0]], label[rel[1]]] += 1.0
    return o2o_m

def get_matrix(ind2class, o2p_m, w2v_filename):
    num_class = len(ind2class)
    #div = np.reshape(np.tile(np.sum(o2p_m, axis=1) + 1, [o2p_m.shape[1]]),
    #                 [o2p_m.shape[1], o2p_m.shape[0]]).transpose(i)
    div=np.sum(o2p_m,axis=1)[:,np.newaxis]+0.00000001
    o2p_m = o2p_m / div
    w2v = np.load(w2v_filename)
    class_v = w2v[:len(ind2class), :]
    oo_v = np.zeros([num_class**2, w2v.shape[1]*2])
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

    #new_o2p_m = np.matmul(o2o_m, o2p_m)

    # new_o2p_m2 = np.matmul(o2o_m, new_o2p_m)
    # new_o2p_m3 = np.matmul(o2o_m, new_o2p_m2)
    # new_o2p_m = new_o2p_m + new_o2p_m2 + new_o2p_m3

    lanta = 0.5
    new_o2p_m = (1-lanta) * np.matmul(np.linalg.inv(np.eye(o2o_m.shape[0], o2o_m.shape[1])-lanta*o2o_m), o2p_m)

    #new_o2p_m = new_o2p_m + o2p_m
    # new_o2p_m = o2p_m

    # rows norm
    new_o2p_m = new_o2p_m/np.sum(new_o2p_m, axis=1)[:,np.newaxis]
    #print(new_o2p_m[55])
    return o2p_m, new_o2p_m

def save(lang_o2p_m, filename, o2p_m=None, d_filename=None):
    np.save(filename, lang_o2p_m)
    if o2p_m is not None:
        np.save(d_filename, o2p_m)
    print("saved successful to {}!".format(filename))

def show_o2p(origin_o2p, o2p, ind2class, ind2predicate, num_show=1000):
    for i in range(len(ind2predicate)):
        while(True):
            # ro = np.random.randint(0, len(origin_o2p))
            # rp = np.random.randint(0, len(origin_o2p[0]))

            # if o2p[ro, rp] > 0.1: break;
            ro, rp = 0*len(ind2class)+41, i
            break
        l1, l2 = reverse_id(ro, len(ind2class))
        if o2p[ro, rp] > 0.01:
            print(ind2class[l1], ind2class[l2], ind2predicate[rp], origin_o2p[ro, rp], o2p[ro, rp])

def vrd_prior(data_dir, suffix=''):
    w2v_filename = './data/{}/w2v_all{}.npy'.format(data_dir, suffix)
    info = json.load(open('./data/{}/train.json'.format(data_dir)))
    o2p_m = get_vrd_o2pm(info)
    origin_o2p, o2p = get_matrix(info['ind_to_class'], o2p_m, w2v_filename)
    save(o2p, "./data/{}/lang_prior{}".format(data_dir, suffix), origin_o2p, "./data/{}/dataset_prior".format(data_dir))
    show_o2p(origin_o2p, o2p, info['ind_to_class'], info['ind_to_predicate'])

    # origin_o2o, o2o = get_vrd_o2o_matrix(info, w2v_filename)
    # print(o2o)
    # save(o2o, "./data/vrd/lang_prior_o2o.pickle", origin_o2o, "./data/vrd/dataset_prior_o2o.pickle")

def vg_prior():
    w2v_filename = './data/vg/w2v_all.npy'
    o2p_m, ind2class, ind2predicate = get_vg_info()
    origin_o2p, o2p = get_matrix(ind2class, o2p_m, w2v_filename)
    #show_o2p(origin_o2p, o2p, ind2class, ind2predicate)
    save(o2p, "./data/vg/lang_prior", origin_o2p, "./data/vg/dataset_prior")

if __name__=='__main__':
    vrd_prior('vrd', '')
