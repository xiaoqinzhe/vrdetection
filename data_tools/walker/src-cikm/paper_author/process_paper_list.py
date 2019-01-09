#coding:utf-8

import os
import json

file_list = set(list(filter(lambda item: item.endswith('.json'), os.listdir('.'))))
print(file_list)

authors = {
    'chong-zhao': 'author5328', 
    'jing-xu': 'author3180', 
    'li-jiang': 'author208', 
    'peng-lu': 'author835', 
    'wei-xue': 'author375', 
    'yong-xiang': 'author435', 
    'yu-long': 'author192',
}

author_papers = {author:{'true':[], 'false':[]} for author in authors}

for author in authors:
    for i in range(6):
        file_name = '{}-{}.json'.format(author, i)
        print(file_name)
        if file_name in file_list:
            with open(file_name) as file:
                papers = json.loads(file.read())
                for paper in papers:
                    if paper['label'] not in ['true', 'false']:
                        continue
                    author_papers[author][paper['label']].append(paper['_id'])

with open('author_paper_label.txt', 'w') as file:
    for author, samples in author_papers.items():
        for paper in samples['true']:
            file.write('{} {} 1\n'.format(authors[author], paper))
        for paper in samples['false']:
            file.write('{} {} 0\n'.format(authors[author], paper))
        