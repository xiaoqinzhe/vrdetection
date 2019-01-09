#coding:utf-8

nodes_file = '../dataset/nodes.txt'
vocab_file = '../dataset/vocab.txt'
venues_file = '../dataset/venues.txt'

with open(vocab_file) as wfile, open(venues_file) as cfile, open(nodes_file, 'w') as ofile:
    for line in wfile:
        word = line.strip()
        ofile.write('{} word\n'.format(word))
    for line in cfile:
        venue = line.strip()
        ofile.write('{} conf\n'.format(venue))
