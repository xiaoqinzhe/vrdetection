#coding:utf-8

import json

with open('../dataset/word.json') as file:
    words = json.loads(file.read())

with open('../dataset/word_category.txt', 'w') as file:
    for i, (cate, wlist) in enumerate(words.items()):
        for word in wlist:
            file.write('{} {}\n'.format(word, i))