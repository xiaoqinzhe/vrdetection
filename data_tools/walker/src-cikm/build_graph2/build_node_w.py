#coding:utf-8

with open('../dataset2/vocab.txt') as ifile, open('../dataset2/node_w.txt', 'w') as ofile:
    for line in ifile:
        w = line.strip()
        ofile.write('{} word\n'.format(w))
