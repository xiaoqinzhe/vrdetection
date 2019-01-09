from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import numpy as np
import json

def save_w2v_from_w2v_all(w2v_all_file, num_classes, saved_file):
    w2v_all = np.load(w2v_all_file)
    np.save(saved_file, w2v_all[:100])

if __name__=='__main__':
    pass
