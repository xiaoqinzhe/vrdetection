#coding:utf-8
import numpy as np

from util import load_graph, HeteroGraphLookup, load_embeddings, load_author_paper_label

# graph = load_graph('dataset2/nodes_all.txt', 'dataset2/edges_all.txt')
# lookup = HeteroGraphLookup(graph)
embeddings = load_embeddings('/home/zjj/Projects/results-compare/hin2vec.txt')
author_paper_label = load_author_paper_label('dataset2/author_paper_label.txt')

with open('results-compare/scores_hin2vec.txt', 'w') as file:
    for author, paper, label in author_paper_label:
        emb_author = embeddings.get(author, np.array([0. for i in range(128)]))
        emb_paper = embeddings.get('!'+paper[1:], np.array([0. for i in range(128)]))
        file.write('{} {} {} {}\n'.format(author, paper, label, np.sum(np.multiply(emb_author, emb_paper))))
