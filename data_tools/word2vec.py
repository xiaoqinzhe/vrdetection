from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import numpy as np
import json

def convert():
    glove2word2vec('data/word2vec/glove/glove.6B.300d.txt', 'data/word2vec/glove.6B.300d.txt')
    print('convert success!')

def get_wordvecs(w2v_file):
    print('loading word vector from %s' % w2v_file)
    if w2v_file.endswith('.txt'):
        wordvec = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    else:
        wordvec = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)
    print('load done.')
    return wordvec

def save_wordvecs(w2v, words, save_file):
    print('getting word embedding...')
    vecs = np.zeros([len(words), len(w2v['the'])])
    for i, word in enumerate(words):
        ws = word.split(' ')[0]
        for w in ws: vecs[i] += w2v[w]
    np.save(save_file, np.array(vecs))
    print('done. save to %s' % save_file)

def get_vrd_words(json_file):
    info = json.load(open(json_file))
    return info['ind_to_class']

if __name__=='__main__':
    w2v = get_wordvecs('./data/word2vec/glove.6B.300d.txt')
    words = get_vrd_words('./data/vrd/train.json')
    save_wordvecs(w2v, words, './data/vrd/w2v')
