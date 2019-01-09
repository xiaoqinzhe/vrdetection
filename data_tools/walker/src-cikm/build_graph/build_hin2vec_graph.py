#coding:utf-8

node_type = {}

with open('../dataset/nodes_all.txt') as file:
    for line in file:
        node, typ = line.strip().split()
        node_type[node] = typ

with open('../dataset/edges_all.txt') as ifile, open('../dataset/edges_hin2vec.txt', 'w') as ofile:
    for node, typ in node_type.items():
        if node.startswith('#'):
            node = '!' + node[1:]
        ofile.write('{}\t{}\t{}\t{}\t{}-{}\n'.format(node, typ[0], '**dummy', 'd', typ[0], 'd'))
        ofile.write('{}\t{}\t{}\t{}\t{}-{}\n'.format('**dummy', 'd', node, typ[0], 'd', typ[0]))
    for line in ifile:
        v1, v2, weight = line.strip().split()
        v1_t = node_type[v1][0]
        v2_t = node_type[v2][0]
        if v1.startswith('#'):
            v1 = '!' + v1[1:]
        if v2.startswith('#'):
            v2 = '!' + v2[1:]
        ofile.write('{}\t{}\t{}\t{}\t{}-{}\n'.format(v1, v1_t, v2, v2_t, v1_t, v2_t))
        ofile.write('{}\t{}\t{}\t{}\t{}-{}\n'.format(v2, v2_t, v1, v1_t, v2_t, v1_t))