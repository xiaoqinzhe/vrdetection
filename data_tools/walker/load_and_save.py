#coding:utf-8

import networkx as nx

def load_edgelist(file_name):
    graph = nx.Graph()
    edges = []
    with open(file_name) as file:
        for line in file:
            if line.startswith('#'):
                continue
            l = line.strip().split()
            if len(l)==2:
                u, v = l
                edges.append((u, v, 1))
            else:
                u, v, weight = l
                edges.append((u, v, float(weight)))

    for u, v, weight in edges:
        graph.add_edge(u, v, weight=weight)

    return graph

def save_embedding(embeddings, file_name):
    with open(file_name, 'w') as file:
        for key, val in embeddings.items():
            file.write('{} {}\n'.format(
                key,
                ' '.join(list(map(str, val))),
            ))
