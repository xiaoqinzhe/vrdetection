#coding:utf-8
import json
from pprint import pprint
paper_author = {}

with open('../dataset/author_paper_label.txt') as file:
    for line in file:
        author, paper, label = line.strip().split()
        if author=='author29':
            break
        if label=='1':
            paper_author[paper] = author

author_paper = {}

with open('../dataset/dblp_ref.json') as file:
    for line in file:
        paper = json.loads(line)
        if paper['_id'] in paper_author:
            author = paper_author[paper['_id']]
            if author not in author_paper:
                author_paper[author] = []
            author_paper[author].append((
                paper.get('title', 'no-title'),
                paper.get('year', 'no-year'),
            ))

for author, papers in author_paper.items():
    print(author, len(papers))
    papers.sort(key=lambda item: item[1], reverse=True)
    for title, year in papers:
        print(year, title)
        