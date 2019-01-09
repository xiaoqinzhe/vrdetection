#coding:utf-8

with open('../dataset/nodes_all.txt') as ifile, open('../dataset/nodes_esim.txt', 'w') as ofile:
    for line in ifile:
        node, typ = line.strip().split()
        ofile.write('{} {}\n'.format(node, typ[0]))