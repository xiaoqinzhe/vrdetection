import networkx as nx
import json
from walker import Walker
import pickle
import gensim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def get_data(dataset='vrd'):
    if dataset == 'vrd':
        data = json.load(open('./data/vrd/train.json'))
    else:
        data = None
    return data

def get_graph(data):
    g = nx.DiGraph()
    for ele in data:
        rels = ele['relations']
        labels = ele['labels']
        for rel in rels:
            o1, o2 = labels[rel[0]], labels[rel[1]]
            if g.has_edge(o1, o2):
                w = g.get_edge_data(o1, o2)['weight'] + 1.0
                g.add_edge(o1, o2, weight=w)
            else: g.add_edge(o1, o2, weight=1.0)
    return g

def get_graph_all(data, num_classes=100):
    g = nx.DiGraph()
    for ele in data:
        rels = ele['relations']
        labels = ele['labels']
        for rel in rels:
            o1, o2 = labels[rel[0]], labels[rel[1]]
            p = num_classes + rel[2]
            for a, b in [[o1, p], [p, o2]]:
                if g.has_edge(a, b):
                    w = g.get_edge_data(a, b)['weight'] + 1.0
                    g.add_edge(a, b, weight=w)
                else: g.add_edge(a, b, weight=1.0)
    return g

def train_obj_emb():
    data = get_data('vrd')
    g = get_graph(data['data'])
    ind2class = [str.replace(" ", "_") for str in data['ind_to_class']]
    class2ind = {}
    for i, cls in enumerate(ind2class): class2ind[cls] = i
    walker = Walker(walk_length=80, multi_process='single', weighted=True)
    seqs = walker.walk(g)
    # print(seqs[:10])
    # pickle.dump(seqs, open('./data/walker/seq_vrd.pickle', 'w'))

    # run skip_gram
    seqs = list(map(
        lambda sequence: list(map(lambda seq: ind2class[seq], sequence)),
        seqs,
    ))
    embedding_size = 50
    temp_model = gensim.models.KeyedVectors.load_word2vec_format('./data_tools/walker/w2v_vrd.50d.txt', binary=False)
    model = gensim.models.Word2Vec(min_count=0, sg=1, window=2, size=embedding_size, iter=0)
    model.build_vocab(seqs)
    model.build_vocab([list(temp_model.vocab.keys())],update=True)
    model.intersect_word2vec_format("./data_tools/walker/w2v_vrd.50d.txt", binary=False, lockf=1.0)
    model.train(seqs, total_examples=model.corpus_count, epochs=model.iter)
    print('word2vec done')
    # res = model.wv
    res = temp_model
    obj_embs = np.zeros([100, embedding_size], dtype=np.float32)
    for cls in class2ind:
        ind = class2ind[cls]
        obj_embs[ind] = res[cls]

    np.save('./data/vrd/w2v_new', obj_embs)
    print("saved to {}".format('./data/vrd/w2v_new' + '.npy'))

    # show data
    tsne = TSNE(n_components=2)
    reduced_emb = tsne.fit_transform(obj_embs)
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1])
    for i in range(len(reduced_emb)):
        plt.annotate(ind2class[i], reduced_emb[i, :], fontsize=6)
    # plt.show()
    plt.savefig('./data_tools/walker/a.png')
    print("plot figure saved in {}".format('./data_tools/walker/a.png'))

def train_words_emb():
    data = get_data('vrd')
    g = get_graph_all(data['data'])
    ind2class = [str.replace(" ", "_") for str in data['ind_to_class']]
    class2ind = {}
    for i, cls in enumerate(ind2class): class2ind[cls] = i
    ind2predicate = [str.replace(" ", "_") for str in data['ind_to_predicate']]
    predicate2ind = {}
    for i, cls in enumerate(ind2predicate): predicate2ind[cls] = i
    ind2name = np.hstack([ind2class, ind2predicate])
    name2ind = {}
    for i, name in enumerate(ind2name): name2ind[name] = i

    walker = Walker(walk_length=80, multi_process='single', weighted=True)
    seqs = walker.walk(g)
    # print(seqs[:10])
    # pickle.dump(seqs, open('./data/walker/seq_vrd.pickle', 'w'))

    # run skip_gram
    seqs = list(map(
        lambda sequence: list(map(lambda seq: ind2name[seq], sequence)),
        seqs,
    ))
    embedding_size = 50
    temp_model = gensim.models.KeyedVectors.load_word2vec_format('./data_tools/walker/w2v_all_vrd.50d.txt', binary=False)
    model = gensim.models.Word2Vec(min_count=0, sg=1, window=2, size=embedding_size, iter=50)
    model.build_vocab(seqs)
    model.build_vocab([list(temp_model.vocab.keys())], update=True)
    model.intersect_word2vec_format("./data_tools/walker/w2v_all_vrd.50d.txt", binary=False, lockf=1.0)
    model.train(seqs, total_examples=model.corpus_count, epochs=model.iter)
    print('word2vec done')
    res = model.wv
    obj_embs = np.zeros([100, embedding_size], dtype=np.float32)
    for cls in class2ind:
        obj_embs[class2ind[cls]] = res[cls]

    # np.save('./data/vrd/w2v_all_new', obj_embs)
    print("saved to {}".format('./data/vrd/w2v_new_all' + '.npy'))

    # show data
    plt.figure()
    tsne = TSNE(n_components=2)
    reduced_emb = tsne.fit_transform(obj_embs)
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1])
    for i in range(len(reduced_emb)):
        plt.annotate(ind2class[i], reduced_emb[i, :])
    plt.savefig('./data_tools/walker/objs.png')
    print("plot figure saved in {}".format('./data_tools/walker/objs.png'))

    plt.figure()
    tsne = TSNE(n_components=2)
    embs = np.zeros([70, embedding_size], dtype=np.float32)
    for cls in predicate2ind:
        embs[predicate2ind[cls]] = res[cls]
    reduced_emb = tsne.fit_transform(embs)
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1])
    for i in range(len(reduced_emb)):
        plt.annotate(ind2predicate[i], reduced_emb[i, :])
    plt.savefig('./data_tools/walker/predicates.png')
    print("plot figure saved in {}".format('./data_tools/walker/predicates.png'))

    plt.figure()
    tsne = TSNE(n_components=2)
    embs = np.zeros([170, embedding_size], dtype=np.float32)
    for cls in name2ind:
        embs[name2ind[cls]] = res[cls]
    reduced_emb = tsne.fit_transform(embs)
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1])
    for i in range(len(reduced_emb)):
        plt.annotate(ind2name[i], reduced_emb[i, :])
    plt.savefig('./data_tools/walker/words.png')
    print("plot figure saved in {}".format('./data_tools/walker/words.png'))

if __name__ == '__main__':
    train_obj_emb()