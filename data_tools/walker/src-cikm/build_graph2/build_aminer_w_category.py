#coding:utf-8

import json

with open('../dataset2/word.json') as file:
    words = json.loads(file.read())

with open('../dataset2/vocab.txt') as file:
    vocab = set([line.strip() for line in file])
with open('../dataset2/word_category.txt', 'w') as file:
    for i, (cate, wlist) in enumerate(words.items()):
        for word in wlist:
            if word in vocab:
                file.write('{} {}\n'.format(word, i))