#coding:utf-8

with open('../dataset/venues.txt') as ifile, open('../dataset/node_c.txt', 'w') as ofile:
    for line in ifile:
        c = line.strip()
        ofile.write('{} conf\n'.format(c))
