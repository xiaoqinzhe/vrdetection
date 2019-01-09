#coding:utf-8

import numpy as np
from util import load_graph, load_embeddings, HeteroGraphLookup
def build_knn_mat(emb_mat, k, min_degrees=None):
    if min_degrees is None:
        min_degrees = np.array([k for i in range(emb_mat.shape[0])])
    fc_mat = np.matmul(emb_mat, np.transpose(emb_mat))
    for i in range(fc_mat.shape[0]):
        row = fc_mat[i,:]
        row[i] = 0
        ik = min_degrees[i]
        k_th_max = np.max([np.partition(row, -ik)[-ik], 1e-5])
        row[row<k_th_max] = 0
    zero_indices = (fc_mat<1e-5).nonzero()
    fc_mat[zero_indices[0], zero_indices[1]] = fc_mat[zero_indices[1], zero_indices[0]]
    return fc_mat
