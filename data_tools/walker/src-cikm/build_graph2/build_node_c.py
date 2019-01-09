#coding:utf-8

with open('../dataset2/venues.txt') as ifile, open('../dataset2/node_c.txt', 'w') as ofile:
    for line in ifile:
        c = line.strip()
        ofile.write('{} conf\n'.format(c))
