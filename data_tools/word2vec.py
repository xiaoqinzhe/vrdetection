from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import numpy as np
import json

def convert(glove_file,  save_file):
    glove2word2vec(glove_file, save_file)
    print('convert success!')

def get_wordvecs(w2v_file):
    print('loading word vector from %s' % w2v_file)
    if w2v_file.endswith('.txt'):
        wordvec = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    else:
        wordvec = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)
    print('load done.')
    return wordvec

def save_wordvecs(w2v, words, save_file, preds=None):
    print('getting word embedding...')
    if preds is not None:
        words = np.hstack([words, preds])
    vecs = np.zeros([len(words), len(w2v['the'])])
    for i, word in enumerate(words):
        ws = word.split(' ')
        for w in ws: vecs[i] += w2v[w]
    np.save(save_file, np.array(vecs))
    print('done. save to %s' % save_file)

def get_vrd_words(json_file):
    info = json.load(open(json_file))
    return info['ind_to_class'], info['ind_to_predicate']

def get_vg_words(json_file):
    info = json.load(open(json_file))
    class_to_ind = info['label_to_idx']
    for name in class_to_ind: class_to_ind[name] -= 1
    ind_to_classes = ['a' for _ in range(len(class_to_ind))]
    for name in class_to_ind: ind_to_classes[class_to_ind[name]] = name

    predicate_to_ind = info['predicate_to_idx']
    for name in predicate_to_ind: predicate_to_ind[name] -= 1
    ind_to_predicates = ['a' for _ in range(len(predicate_to_ind))]
    for name in predicate_to_ind: ind_to_predicates[predicate_to_ind[name]] = name

    return ind_to_classes, ind_to_predicates

def show_embedding(ind2class, ind2predicate, filename):
    w2v = np.load(filename)
    class_v = w2v[:len(ind2class), :]
    predicate_v = w2v[len(ind2class):, :]

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    def viz(vec, ind2name, savename):
        tsne = TSNE(2)
        nv = tsne.fit_transform(vec)
        plt.figure()
        plt.scatter(nv[:,0], nv[:,1])
        for i in range(len(vec)):
            plt.annotate(ind2name[i], nv[i], fontsize=8)
        plt.savefig(savename)

    viz(class_v, ind2class, "./data_tools/fig/obj_emb.png")
    viz(predicate_v, ind2predicate, "./data_tools/fig/pred_emb.png")

if __name__=='__main__':
    # # convert('data/word2vec/glove/glove.6B.50d.txt', 'data/word2vec/glove.6B.50d.txt')
    dataset_name = 'vg/vg_vtranse'
    w2v = get_wordvecs('./data/word2vec/glove.6B.50d.txt')
    objs, preds = get_vrd_words('./data/{}/train.json'.format(dataset_name))
    save_wordvecs(w2v, objs, './data/{}/w2v'.format(dataset_name))
    save_wordvecs(w2v, objs, './data/{}/w2v_all'.format(dataset_name), preds=preds)

    # objs, preds = get_vrd_words('./data/vrd/train.json')
    # show_embedding(objs, preds, './data/vrd/w2v_all.npy')

    # vg dataset
    # w2v = get_wordvecs('./data/word2vec/glove.6B.50d.txt')
    # objs, preds = get_vg_words('./data/vg/VG-dicts.json')
    # save_wordvecs(w2v, objs, './data/vg/w2v')
    # save_wordvecs(w2v, objs, './data/vg/w2v_all', preds=preds)
    # show_embedding(objs, preds, './data/vg/w2v_all.npy')
