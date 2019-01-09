#coding:utf-8
import json
from pprint import pprint
paper_author = {}

with open('../dataset2/author_paper_label.txt') as file:
    for line in file:
        author, paper, label = line.strip().split()

        if label=='1':
            paper_author[paper] = author

author_paper = {}

with open('../dataset2/dblp_ref.json') as file:
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
    last_year = -1
    for title, year in papers:
        if year!=last_year:
            print()
        last_year=year
        print(year, title)
        