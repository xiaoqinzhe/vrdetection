#coding:utf-8
import os
from pprint import pprint
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from util import (
    load_graph,
    HeteroGraphLookup,
    load_labels,
    save_embeddings,
    load_embeddings,
    embeddings_arr_to_dict,
    load_author_paper_label,
    save_author_paper_label_score,
    local_network_degrees,
)
from walker import Walker, TransProbCalculator
from visualize import plot_tsne
from emb_graph import build_knn_mat
from model_lr import Model

os.environ['CUDA_VISIBLE_DEVICES']='1'

conf_labels = load_labels('vrd/conf_category.txt')
word_labels = load_labels('vrd/word_category.txt')
print('Loading graph...')
origin_graph = load_graph('vrd/nodes.txt', 'vrd/edges.txt')
print('Building lookups...')
lookup = HeteroGraphLookup(origin_graph)

graph = nx.convert_node_labels_to_integers(origin_graph)

mat = nx.to_numpy_matrix(graph)
mat = normalize(mat)
# mat = np.matmul(mat, np.eye(mat.shape[0])+np.matmul(mat, np.eye(mat.shape[0])+mat))/3
mat = np.matmul(mat, np.eye(mat.shape[0])+mat)/2

cls_mat = normalize(mat[:100, :100])
pred_mat = normalize(mat[100:, 100:])

walker = Walker(graph)
calculator = TransProbCalculator(graph, lookup)

seq = walker.walk(num_walks=16)

print('Training conf...')
# conf_mat = calculator.cal_prob_mat('0', seq)
# np.save('results-new/conf.npy', arr=conf_mat)
# conf_mat = np.load('results-new/conf.npy')
conf_model = Model(cls_mat, embedding_dim=16, iterations=200, learning_rate=0.001, batch_size=10, name_scope='conf')
conf_emb = conf_model.train().get_embeddings()

conf_emb2=np.zeros_like(conf_emb)
ind2conf=[' ' for i in range(len(conf_emb))]
for label, idx in conf_labels.items():
    conf_emb2[idx] = conf_emb[lookup.node_label_to_type_index(str(idx))]
    ind2conf[idx]=label

print('Computing conf tsne...')
from sklearn.manifold import TSNE
plt.figure()
tsne = TSNE(n_components=2)
reduced_emb = tsne.fit_transform(conf_emb2)
plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1])
for i in range(len(reduced_emb)):
    plt.annotate(ind2conf[i], reduced_emb[i, :])
plt.savefig('./results/objs.png')

print('Training word...')
# word_mat = calculator.cal_prob_mat('1', seq)
# np.save('results-new/word.npy', arr=word_mat)
# word_mat = np.load('results-new/word.npy')
word_model = Model(pred_mat, embedding_dim=16, iterations=200, learning_rate=0.005, batch_size=100, name_scope='word')
word_emb = word_model.train().get_embeddings()


word_emb2=np.zeros_like(word_emb)
ind2word=[' ' for i in range(len(word_emb))]
for label, idx in word_labels.items():
    word_emb2[idx-100] = word_emb[lookup.node_label_to_type_index(str(idx))]
    ind2word[idx-100]=label

print('Computing conf tsne...')
from sklearn.manifold import TSNE
plt.figure()
tsne = TSNE(n_components=2)
reduced_emb = tsne.fit_transform(word_emb2)
plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1])
for i in range(len(reduced_emb)):
    plt.annotate(ind2word[i], reduced_emb[i, :])
plt.savefig('./results/predicates.png')

print('Computing word tsne...')
plot_tsne({label:word_emb[lookup.node_label_to_type_index(str(idx)),:] for label, idx in word_labels.items()}, word_labels, 'results/word.png')

emb=np.vstack((conf_emb2, word_emb2))
np.save('../../../data/vrd/w2v_all_hete', emb)
print('saved embedding to w2v_all_hete.npy')