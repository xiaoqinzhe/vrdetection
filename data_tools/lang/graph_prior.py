import os, sys
this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, '..'))

import networkx as nx
import json
from walker.walker import Walker
from walker.opowalker import opo_walk
import pickle
import gensim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.switch_backend('agg')

np.random.seed(666)

def get_data(dataset='vrd'):
    if dataset == 'vrd':
        data = json.load(open('./data/vrd/train.json'))
    else:
        data = None
    return data

def build_graph(data, num_classes=100):
    g = nx.DiGraph()
    for ele in data:
        rels = ele['relations']
        labels = ele['labels']
        for rel in rels:
            o1, o2 = labels[rel[0]], labels[rel[1]]
            p = rel[2] + num_classes
            if g.has_edge(o1, p):
                w = g.get_edge_data(o1, p)['weight'] + 1.0
                g.add_edge(o1, p, weight=w)
                # pass
            else:  g.add_edge(o1, p, weight=1.0)
            if g.has_edge(p, o2):
                w = g.get_edge_data(p, o2)['weight'] + 1.0
                g.add_edge(p, o2, weight=w)
                # pass
            else:  g.add_edge(p, o2, weight=1.0)
    return g

def train_words_emb():
    data = get_data('vrd')
    g = build_graph(data['data'])
    ind2class = [str.replace(" ", "_") for str in data['ind_to_class']]
    class2ind = {}
    for i, cls in enumerate(ind2class): class2ind[cls] = i
    ind2predicate = [str.replace(" ", "_") for str in data['ind_to_predicate']]
    predicate2ind = {}
    for i, cls in enumerate(ind2predicate): predicate2ind[cls] = i
    ind2name = np.hstack([ind2class, ind2predicate])
    name2ind = {}
    for i, name in enumerate(ind2name): name2ind[name] = i

    embedding_size = 16

    walker_way = '1'
    walker = Walker(num_walks=10, walk_length=80, multi_process='single', weighted=True)
    seqs = walker.walk(g)

    # walker_way = '2'
    # g, seqs = opo_walk('./data_tools/walker/src-cikm/vrd/nodes.txt', './data_tools/walker/src-cikm/vrd/edges.txt')

    saved_filename = './data/vrd/w2v_all_graph_{}_{}'.format(walker_way, embedding_size)

    # print(seqs[:10])
    # pickle.dump(seqs, open('./data/walker/seq_vrd.pickle', 'w'))

    # run skip_gram
    seqs = list(map(
        lambda sequence: list(map(lambda seq: ind2name[int(seq)], sequence)),
        seqs,
    ))

    # no initial w2v weights
    model = gensim.models.Word2Vec(seqs, min_count=0, sg=1, hs=1, window=2, size=embedding_size, iter=200)
    # use w2v weights
    # temp_model = gensim.models.KeyedVectors.load_word2vec_format('./data_tools/walker/w2v_all_vrd.50d.txt', binary=False)
    # model = gensim.models.Word2Vec(min_count=0, sg=1, window=2, size=embedding_size, iter=50)
    # model.build_vocab(seqs)
    # # model.build_vocab([list(temp_model.vocab.keys())], update=True)
    # model.intersect_word2vec_format("./data_tools/walker/w2v_all_vrd.50d.txt", binary=False, lockf=1.0)
    # model.train(seqs, total_examples=model.corpus_count, epochs=model.iter)
    print('word2vec done')
    res = model.wv
    obj_embs = np.zeros([100, embedding_size], dtype=np.float32)

    www=np.load('./data/vrd/w2v_all.npy')
    for cls in class2ind:
        obj_embs[class2ind[cls]] = res[cls]
        # print(cls, res[cls])
        # print(www[class2ind[cls]])

    np.save(saved_filename, obj_embs)
    print("saved to {}".format(saved_filename))

    # show data
    plt.figure()
    tsne = TSNE(n_components=2)
    reduced_emb = tsne.fit_transform(obj_embs)
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1])
    for i in range(len(reduced_emb)):
        plt.annotate(ind2class[i], reduced_emb[i, :])
    plt.savefig('./data_tools/fig/objs_{}_{}.png'.format(walker_way, embedding_size))
    print("plot figure saved in {}".format('./data_tools/fig/objs.png'))

    plt.figure()
    tsne = TSNE(n_components=2)
    embs = np.zeros([70, embedding_size], dtype=np.float32)
    for cls in predicate2ind:
        embs[predicate2ind[cls]] = res[cls]
    reduced_emb = tsne.fit_transform(embs)
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1])
    for i in range(len(reduced_emb)):
        plt.annotate(ind2predicate[i], reduced_emb[i, :])
    plt.savefig('./data_tools/fig/predicates.png')
    print("plot figure saved in {}".format('./data_tools/fig/predicates.png'))

    plt.figure()
    tsne = TSNE(n_components=2)
    embs = np.zeros([170, embedding_size], dtype=np.float32)
    for cls in name2ind:
        embs[name2ind[cls]] = res[cls]
    reduced_emb = tsne.fit_transform(embs)
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1])
    for i in range(len(reduced_emb)):
        plt.annotate(ind2name[i], reduced_emb[i, :])
    plt.savefig('./data_tools/fig/words.png')
    print("plot figure saved in {}".format('./data_tools/fig/words.png'))

if __name__ == '__main__':
    train_words_emb()