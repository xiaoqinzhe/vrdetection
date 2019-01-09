#coding:utf-8

with open('../dataset/vocab.txt') as ifile, open('../dataset/node_w.txt', 'w') as ofile:
    for line in ifile:
        w = line.strip()
        ofile.write('{} word\n'.format(w))
