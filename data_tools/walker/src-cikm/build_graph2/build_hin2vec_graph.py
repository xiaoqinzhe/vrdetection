#coding:utf-8

node_type = {}

with open('../dataset2/nodes_all.txt') as file:
    for line in file:
        node, typ = line.strip().split()
        node_type[node] = typ

with open('../dataset2/edges_deepwalk.txt') as ifile, open('../dataset2/edges_hin2vec_homo.txt', 'w') as ofile:
    for line in ifile:
        v1, v2, weight = line.strip().split()
        # v1_t = node_type[v1][0]
        # v2_t = node_type[v2][0]
        v1_t = 'n'
        v2_t = 'n'
        if v1.startswith('#'):
            v1 = '!' + v1[1:]
        if v2.startswith('#'):
            v2 = '!' + v2[1:]
        ofile.write('{}\t{}\t{}\t{}\t{}-{}\n'.format(v1, v1_t, v2, v2_t, v1_t, v2_t))
        ofile.write('{}\t{}\t{}\t{}\t{}-{}\n'.format(v2, v2_t, v1, v1_t, v2_t, v1_t))