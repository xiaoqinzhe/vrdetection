#coding:utf-8
import os
vocab_add_file_name = '../dataset/vocab_add.txt'
vocab_remove_file_name = '../dataset/vocab_remove.txt'

vocab_add = set()
vocab_remove = set()

with open(vocab_add_file_name) as file:
    for line in file:
        vocab_add.add(line.strip())

with open(vocab_remove_file_name) as file:
    for line in file:
        vocab_remove.add(line.strip())

vocab = vocab_add - vocab_remove

vocab = list(vocab)
vocab.sort()

with open('../dataset/vocab.txt', 'w') as file:
    for word in vocab:
        file.write('{}\n'.format(word))

os.system('cp ../dataset/edges_ww_final.txt ../dataset/edges_ww.txt')
