#coding:utf-8
import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from util import load_graph, HeteroGraphLookup, load_aminer_conf, save_embeddings
from walker import Walker, TransProbCalculator
from visualize import plot_tsne


aminer_conf = load_aminer_conf()

print('Loading graph...')
graph = load_graph('dataset/nodes.txt', 'dataset/edges.txt')
print('Building lookups...')
lookup = HeteroGraphLookup(graph)

for i in range(10):
    label = lookup.node_type_index_to_label('conf', i)
    print(i, label)
    g_id = lookup.node_type_index_to_global_index('conf', i)
    print(g_id, lookup.node_index_to_label(g_id))
    print(g_id, lookup.node_global_index_to_type_index(g_id))