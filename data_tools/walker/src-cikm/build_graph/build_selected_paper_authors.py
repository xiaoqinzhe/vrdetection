#coding:utf-8
import json
import re

re_non_letter_space = re.compile(r'[^A-Za-z ]+')
re_spaces = re.compile(r' +')

def process_name(name):
    name = re_non_letter_space.sub('', name)
    name = re_spaces.sub('-', name)
    name = name.lower()
    return name

papers = set()
with open('../dataset/author_paper_label.txt') as file:
    for line in file:
        author, paper, label = line.strip().split()
        papers.add(paper)

with open('../dataset/dblp_ref.json') as ifile, open('../dataset/selected_paper_authors.txt', 'w') as ofile:
    for line in ifile:
        paper = json.loads(line)
        if paper['_id'] in papers:
            ofile.write('{} {}\n'.format(paper['_id'], ' '.join([process_name(name) for name in paper['authors']])))
