#coding:utf-8
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from util import load_graph, HeteroGraphLookup

def plot_tsne(embeddings, labels, save_file):
    emb_arr = []
    label_arr = []
    for node in labels:
        emb_arr.append(embeddings[node])
        label_arr.append(labels[node])

    emb_arr = np.array(emb_arr)
    label_arr = np.array(label_arr)
    label_s = set(label_arr)
    cmap = plt.cm.get_cmap(name='spectral', lut=len(label_s))

    tsne_result = TSNE(n_components=2).fit_transform(emb_arr)
    lo = np.min(tsne_result)
    up = np.max(tsne_result)
    plt.xlim(lo-10, up+10)
    plt.ylim(lo-10, up+10)
    groups = []
    legends = []
    for i, label in enumerate(label_s):
        selected = tsne_result[label_arr==label]
        x1 = selected[:,0]
        x2 = selected[:,1]
        gr = plt.scatter(x1, x2, c=cmap(i), s=5)
        groups.append(gr)
        legends.append(label)
    plt.legend(groups, legends)

    plt.savefig(save_file)
    plt.clf()

if __name__=='__main__':
    print('Loading...')
    mat = np.load('results/emb_conf.npy')
    graph = load_graph('dataset/nodes.txt', 'dataset/edges.txt')
    lookup = HeteroGraphLookup(graph)
    with open('dataset/conf.json') as file:
        conf = json.loads(file.read())
    
    print('Building groups...')
    groups = []
    for categ, conflist in conf.items():
        conf_id = [lookup.node_label_to_type_index(label) for label in conflist]
        groups.append(mat[conf_id,:])


    print('Calculating tsne...')
    plot_tsne(groups, 'venue', 'conf.png')
        
