#coding:utf-8
import uuid
import json
import networkx as nx
import numpy as np
def random_file_name(prefix, postfix):
    return '{}_{}.{}'.format(prefix, str(uuid.uuid4().hex), postfix)

class HeteroGraphLookup(object):

    def __init__(self, graph):
        self.node_index2label = list(graph.nodes)
        self.node_label2index= {node:i for i, node in enumerate(self.node_index2label)}
        self.node_label2type = {node_label:node_attr['node_type'] for node_label, node_attr in graph.nodes(data=True)}
        self.all_node_types = set(self.node_label2type.values())
        self.type2labels = {typ:[] for typ in self.all_node_types}
        self.node_label2typeindex = {}
        for node_label, typ in self.node_label2type.items():
            self.node_label2typeindex[node_label] = len(self.type2labels[typ])
            self.type2labels[typ].append(node_label)
        self.node_typeindex2label = {typ:{} for typ in self.all_node_types}
        for label, type_index in self.node_label2typeindex.items():
            typ = self.node_label2type[label]
            self.node_typeindex2label[typ][type_index] = label

    def node_label_to_index(self, label):
        return self.node_label2index[label]
    
    def node_index_to_label(self, index):
        return self.node_index2label[index]

    def node_label_to_type(self, label):
        return self.node_label2type[label]

    def node_label_to_type_index(self, label):
        return self.node_label2typeindex[label]

    def node_index_to_type(self, index):
        label = self.node_index2label[index]
        return self.node_label2type[label]

    def node_type_index_to_global_index(self, node_type, type_index):
        label = self.node_typeindex2label[node_type][type_index]
        return self.node_label2index[label]

    def node_type_index_to_label(self, node_type, type_index):
        return self.node_typeindex2label[node_type][type_index]

    def node_global_index_to_type_index(self, global_index):
        label = self.node_index2label[global_index]
        return self.node_label2typeindex[label]

def load_graph(node_file_name, edge_file_name):
    g = nx.DiGraph()

    with open(node_file_name) as file:
        for line in file:
            node, node_type = line.strip().split()
            g.add_node(node, node_type=node_type)

    with open(edge_file_name) as file:
        for line in file:
            sp = line.strip().split()
            if len(sp)==3:
                u, v, weight = sp
                weight = float(weight)
            else:
                u, v = sp
                weight = 1.
            g.add_edge(u, v, weight=weight)

    return g

def load_labels(label_file):
    labels = {}
    with open(label_file) as file:
        for line in file:
            item, label = line.strip().split()
            label = int(label)
            labels[item] = label

    return labels

def embeddings_arr_to_dict(embeddings, lookup, node_type):
    emb_dict = {}
    for i in range(embeddings.shape[0]):
        label = lookup.node_type_index_to_label(node_type, i)
        emb_dict[label] = embeddings[i,:]
    return emb_dict

def save_embeddings(save_file, embeddings):
    with open(save_file, 'w') as file:
        for label, emb in embeddings.items():
            file.write('{} {}\n'.format(label, ' '.join([str(x) for x in emb])))

def load_embeddings(emb_file):
    embeddings = {}
    with open(emb_file) as file:
        for line in file:
            sp = line.strip().split()
            if len(sp)<3:
                continue
            label = sp[0]
            emb = np.array([float(x) for x in sp[1:]])
            embeddings[label] = emb
    return embeddings

def load_author_paper_label(author_paper_label_file):
    l = []
    with open(author_paper_label_file) as file:
        for line in file:
            author, paper, label = line.strip().split()
            author = '@'+author
            paper = '#'+paper
            label = int(label)
            l.append((author, paper, label))
    
    return l

def save_author_paper_label_score(save_file, author_paper_label, scores):
    with open(save_file, 'w') as file:
        for i, (author, paper, label) in enumerate(author_paper_label):
            score = scores[i]
            file.write('{} {} {} {}\n'.format(author, paper, label, score))

def local_network_degrees(graph, node_type, lookup):
    nodes = set([node for node in graph.nodes if lookup.node_index_to_type(node)==node_type])
    degrees = []
    for i in range(len(nodes)):
        u = lookup.node_type_index_to_global_index(node_type, i)
        neighbors = set(graph.neighbors(u))
        degrees.append(len(neighbors&nodes))
    return np.array(degrees)
