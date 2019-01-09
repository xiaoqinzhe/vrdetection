#coding:utf-8

import json

with open('../dataset/conf.json') as file:
    confs = json.loads(file.read())

with open('../dataset/conf_category.txt', 'w') as file:
    for i, (cate, clist) in enumerate(confs.items()):
        for conf in clist:
            file.write('{} {}\n'.format(conf, i))