#coding:utf-8

import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize

class GraphLookup(object):

    def __init__(self, graph):
        self._index2label = list(graph.nodes)
        self._label2index = {node:i for i, node in enumerate(self._index2label)}
    
    def label_to_index(self, label):
        return self._label2index[label]

    def index_to_label(self, index):
        return self._index2label[index]

def load_opo(nodes_file, edges_file):
    g = nx.DiGraph()

    with open(nodes_file) as file:
        for line in file:
            node, node_type = line.strip().split()
            g.add_node(node, tag=node_type)

    with open(edges_file) as file:
        for line in file:
            o1, p, o2 = line.strip().split()
            if not g.has_edge(o1, p):
                g.add_edge(o1, p, next=set(), weight=0.)
            if not g.has_edge(p, o2):
                g.add_edge(p, o2, last=set(), weight=0.)
            g[o1][p]['weight'] += 1.
            g[p][o2]['weight'] += 1.
            g[o1][p]['next'].add(o2)
            g[p][o2]['last'].add(o1)
    
    lookup = GraphLookup(g)
    g2 = nx.convert_node_labels_to_integers(g)

    for u, v in g2.edges:
        if g2.nodes[u]['tag']=='o':
            key = 'next'
        else:
            key = 'last'
        g2[u][v][key] = {lookup.label_to_index(name) for name in g2[u][v][key]}

    return g, g2, lookup

def make_next_node_table(graph):
    table = {}

    for u in graph.nodes:
        neighbors = list(graph.neighbors(u))
        probs = np.array([graph[u][v]['weight'] for v in neighbors])
        table[u] = {
            'nodes': neighbors,
            'probs': probs/np.sum(probs),
        }

    for u, v in graph.edges:
        if graph.nodes[v]['tag']!='p':
            continue
        # print(u, v, graph.nodes[u]['tag'], graph.nodes[v]['tag'])
        neighbors = list(graph[u][v]['next'])
        probs = np.array([graph[v][w]['weight'] for w in neighbors])
        table[(u,v)] = {
            'nodes': neighbors,
            'probs': probs/np.sum(probs),
        }

    return table

def walk(graph, table, num_walks=10, walk_length=80):
    sequences = []
    for u in graph.nodes:
        for walk in range(num_walks):
            v = np.random.choice(table[u]['nodes'], p=table[u]['probs'])
            seq = [u, v]
            prev = u
            curr = v
            for i in range(walk_length-1):
                if graph.nodes[curr]['tag']=='p':
                    next_ = np.random.choice(table[(prev,curr)]['nodes'], p=table[(prev,curr)]['probs'])
                else:
                    next_ = np.random.choice(table[curr]['nodes'], p=table[curr]['probs'])
                seq.append(next_)
                prev = curr
                curr = next_
            sequences.append(seq)
    return sequences

def opo_walk(node_file, edge_file):
    original_g, g, lookup = load_opo(node_file, edge_file)
    table = make_next_node_table(g)
    sequences = walk(g, table)
    seqs2 = []
    for seq in sequences:
        seqs2.append(list(map(lambda u: lookup.index_to_label(u), seq)))
    return original_g, seqs2
