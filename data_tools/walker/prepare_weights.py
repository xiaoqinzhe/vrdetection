

def save_w2vfile(w2v, ind_to_class, filename):
    with open(filename, 'w') as f:
        f.write(("%d %d\n" % (len(w2v), len(w2v[0]))).encode('utf-8'))
        for i, vec in enumerate(w2v):
            st = ind_to_class[i].replace(" ", "_")
            for i in range(len(vec)): st = st + " " + str(vec[i])
            f.write((st+'\n').encode('utf-8'))
    f.close()
    print('saved to {}'.format(filename))

if __name__ == '__main__':
    import json, numpy as np
    w2v = np.load('./data/vrd/w2v_all.npy')
    data = json.load(open('./data/vrd/train.json'))
    ind2name = np.hstack([data['ind_to_class'], data['ind_to_predicate']])
    save_w2vfile(w2v, ind2name, './data_tools/walker/w2v_all_vrd.50d.txt')