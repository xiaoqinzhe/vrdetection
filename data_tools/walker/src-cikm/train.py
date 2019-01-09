#coding:utf-8
import os

import numpy as np
from model import Model

os.environ['CUDA_VISIBLE_DEVICES']='0'

# mat = np.load('results/conf.npy')
# model = Model(mat, embedding_dim=128, neg_ratio=5, name_scope='conf')
# model.build_computational_graph()
# model.train()
# embeddings = model.get_embeddings()
# np.save('results/emb_conf.npy', arr=embeddings)

# mat = np.load('results/word.npy')
# model = Model(mat, embedding_dim=128, neg_ratio=3, name_scope='word')
# model.build_computational_graph()
# model.train()
# embeddings = model.get_embeddings()
# np.save('results/emb_word.npy', arr=embeddings)

mat = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
mat = mat.reshape((4,4))
mat = np.apply_along_axis(lambda row:row/np.sum(row), axis=1, arr=mat)
print(mat)

model = Model(mat)

print(model.prob_to_weight(mat))