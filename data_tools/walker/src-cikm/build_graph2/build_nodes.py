#coding:utf-8

nodes_file = '../dataset2/nodes.txt'
vocab_file = '../dataset2/vocab.txt'
venues_file = '../dataset2/venues.txt'

with open(vocab_file) as wfile, open(venues_file) as cfile, open(nodes_file, 'w') as ofile:
    for line in wfile:
        word = line.strip()
        ofile.write('{} word\n'.format(word))
    for line in cfile:
        venue = line.strip()
        ofile.write('{} conf\n'.format(venue))
