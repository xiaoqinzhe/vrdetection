import numpy as np

a = np.load("./data/vrd/gp/good_top1_1.npy", allow_pickle=True)

ids = np.argsort(np.array([[d[0], d[1]] for d in a[1]])[:,0])
# ids = np.argsort(-a[1])

print(a[1][ids[-3:]], a[0][ids[-3:]])