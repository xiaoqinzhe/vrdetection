#coding:utf-8
node_mapping = {}

def change_node_name(node, i):
    # if node.startswith('#'):
    #     n = 'p{}'.format(i)
    # elif node.startswith('$'):
    #     n = 'c{}'.format(i)
    # elif node.startswith('@'):
    #     n = 'a{}'.format(i)
    # else:
    #     n = 'w{}'.format(i)

    # return n
    return node
with open('../dataset2/nodes_all.txt') as ifile, open('../dataset2/nodes_esim.txt', 'w') as ofile:
    for i, line in enumerate(ifile):
        node, typ = line.strip().split()
        ofile.write('{} {}\n'.format(change_node_name(node, i), typ[0]))
        node_mapping[node] = change_node_name(node, i)

with open('../dataset2/edges_deepwalk.txt') as ifile, open('../dataset2/edges_esim.txt', 'w') as ofile:
    for line in ifile:
        v1, v2, weight = line.strip().split()
        ofile.write('{} {}\n'.format(node_mapping[v1], node_mapping[v2]))
