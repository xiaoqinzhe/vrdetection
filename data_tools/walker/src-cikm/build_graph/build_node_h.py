#coding:utf-8

authors = set()
with open('../dataset/author_paper_label.txt') as file:
    for line in file:
        author, paper, label = line.strip().split()
        authors.add(author)

with open('../dataset/node_h.txt', 'w') as file:
    for author in authors:
        file.write('@{} homepage\n'.format(author))
