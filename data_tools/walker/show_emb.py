import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
plt.switch_backend('agg')

vrd = json.load(open('../../data/vrd/test.json'))
ind2class=vrd['ind_to_class']
ind2pred = vrd['ind_to_predicate']
ind2name = np.concatenate((ind2class, ind2pred))

embs = np.load('./embedding.npy')
lookup=json.load(open('./lookup.json'))

obj_embs = np.zeros([100, 16], np.float32)
for i in range(100): obj_embs[i]=embs[lookup[str(i)]]

plt.figure()
tsne = TSNE(n_components=2)
reduced_emb = tsne.fit_transform(obj_embs)
plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1])
for i in range(len(reduced_emb)):
    plt.annotate(ind2class[i], reduced_emb[i, :])
plt.savefig('./objs_{}.png'.format(16))
print("plot figure saved in {}".format('./objs.png'))
