#coding:utf-8
import os
from pprint import pprint
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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
from model import Model

os.environ['CUDA_VISIBLE_DEVICES']='0'
num_stages = 5

def replace_emb_edge(graph, node_type, mat):
    nonzero_indices = np.transpose(mat.nonzero())
    edges_to_remove = []
    for u in graph.nodes:
        if lookup.node_index_to_type(u)!=node_type:
            continue
        for v in graph.neighbors(u):
            if lookup.node_index_to_type(v)!=node_type:
                continue
            edges_to_remove.append((u,v))

    graph.remove_edges_from(edges_to_remove)

    for i in range(nonzero_indices.shape[0]):
        u = nonzero_indices[i,0]
        v = nonzero_indices[i,1]
        ug = lookup.node_type_index_to_global_index(node_type, u)
        vg = lookup.node_type_index_to_global_index(node_type, v)
        graph.add_edge(ug, vg, weight=mat[u,v])
    


conf_labels = load_labels('dataset2/conf_category.txt')
word_labels = load_labels('dataset2/word_category.txt')
author_paper_labels = load_author_paper_label('dataset2/author_paper_label.txt')
print('Loading graph...')
origin_graph = load_graph('dataset2/nodes_all.txt', 'dataset2/edges_all.txt')
print('Building lookups...')
lookup = HeteroGraphLookup(origin_graph)

graph = nx.convert_node_labels_to_integers(origin_graph)
# conf_graph = graph
# word_graph = graph

for stage in range(1, num_stages+1):

    # walker = Walker(conf_graph)
    # calculator = TransProbCalculator(conf_graph, lookup)
    walker = Walker(graph)
    calculator = TransProbCalculator(graph, lookup)
    print('Walking paper scores...')
    walk_scores = []
    for author, paper, label in author_paper_labels:
        u_author = lookup.node_label_to_index(author)
        v_paper = lookup.node_label_to_index(paper)
        score = walker.walk_score(u_author, v_paper)
        walk_scores.append(score)
    save_author_paper_label_score('results-2-multistage/walk_scores_stage_{}.txt'.format(stage), author_paper_labels, walk_scores)
    
    seq = walker.walk(num_walks=10)

    print('Stage {} Training conf...'.format(stage))
    conf_mat = calculator.cal_prob_mat('conf', seq)
    conf_model = Model(conf_mat, embedding_dim=128, iterations=100, neg_ratio=0.002, name_scope='stage_{}_conf'.format(stage))
    conf_model.build_computational_graph()
    conf_model.train()
    conf_emb = conf_model.get_embeddings()

    save_embeddings('results-2-multistage/emb_conf_stage_{}.txt'.format(stage), embeddings_arr_to_dict(conf_emb, lookup=lookup, node_type='conf'))

    print('Stage {} Computing conf tsne...'.format(stage))
    plot_tsne({label:conf_emb[lookup.node_label_to_type_index(label),:] for label in conf_labels}, conf_labels, 'results-2-multistage/conf-stage-{}.png'.format(stage))


    print('stage {} Training word...'.format(stage))

    # walker = Walker(word_graph)

    # calculator = TransProbCalculator(word_graph, lookup)
    word_mat = calculator.cal_prob_mat('word', seq)
    word_model = Model(word_mat, embedding_dim=128, iterations=150, neg_ratio=0.0001, learning_rate=0.02, name_scope='stage_{}_word'.format(stage))
    word_model.build_computational_graph()
    word_model.train()
    word_emb = word_model.get_embeddings()

    save_embeddings('results-2-multistage/emb_word_stage_{}.txt'.format(stage), embeddings_arr_to_dict(word_emb, lookup=lookup, node_type='word'))

    print('Stage {} Computing word tsne...'.format(stage))
    plot_tsne({label:word_emb[lookup.node_label_to_type_index(label),:] for label in word_labels}, word_labels, 'results-2-multistage/word-stage-{}.png'.format(stage))



 
    if stage==num_stages:
        break

    print('Building next stage embedding graph...')

    conf_knn_mat = build_knn_mat(conf_emb, k=100, min_degrees=local_network_degrees(graph, 'conf', lookup))
    word_knn_mat = build_knn_mat(word_emb, k=100, min_degrees=local_network_degrees(graph, 'word', lookup))

    # word_graph = nx.Graph(graph)
    # replace_emb_edge(word_graph, 'conf', conf_knn_mat)
    # conf_graph = nx.Graph(graph)
    # replace_emb_edge(conf_graph, 'word', word_knn_mat)
    replace_emb_edge(graph, 'conf', conf_knn_mat)
    replace_emb_edge(graph, 'word', word_knn_mat)
