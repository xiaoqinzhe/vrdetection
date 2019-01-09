#coding:utf-8

import networkx as nx

max_num_edges = 2000000

vocab_file_name = '../dataset2/vocab_add.txt'
paper_title_venue_file_name = '../dataset2/paper_title_venue.txt'

vocab = set()
word_cooc = {}
degree = {}
with open(vocab_file_name) as file:
    for line in file:
        vocab.add(line.strip())

print('Reading...')
with open(paper_title_venue_file_name) as file:
    for line in file:
        paper_id, title, venue = line.strip().split()
        words = title.split('-')

        for i in range(len(words)):
            for j in range(i+1, len(words)):
                w1 = words[i]
                w2 = words[j]
                if w1 in vocab and w2 in vocab:
                    if w1 not in degree:
                        degree[w1] = 0
                    if w2 not in degree:
                        degree[w2] = 0
                    if (w1,w2) not in word_cooc:
                        word_cooc[(w1,w2)] = 0
                        word_cooc[(w2,w1)] = 0
                    word_cooc[(w1,w2)] += 1
                    word_cooc[(w2,w1)] += 1
                    degree[w1] += 1
                    degree[w2] += 1
printed = set()

print('Output to file...')
with open('../dataset2/edges_ww_add.txt', 'w') as file:
    for (w1, w2), cnt in word_cooc.items():
        if (w1,w2) in printed:
            continue
        printed.add((w1,w2))
        printed.add((w2,w1))
        file.write('{} {} {}\n'.format(w1, w2, cnt))
