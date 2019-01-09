#coding:utf-8
import random

author_papers = {}

with open('../dataset/author_paper_label.txt') as file:
    for line in file:
        author, paper, label = line.strip().split()
        if author not in author_papers:
            author_papers[author] = []
        if label=='1':
            author_papers[author].append(paper)
papers = {}

with open('../dataset/paper_selected.txt') as file:
    for line in file:
        p_id, title, venue = line.strip().split()
        papers[p_id] = (title, venue)


author_title = {}
author_venue = {}
for author, apapers in author_papers.items():
    if author not in author_title:
        author_title[author] = []
        author_venue[author] = []
    for paper in apapers:
        title, venue = papers[paper]
        author_title[author].append(title)
        author_venue[author].append(venue)

for author in author_title:
    titles = author_title[author]
    random.shuffle(titles)
    num_samples = max(2, len(titles)//10)
    author_title[author] = titles[:num_samples]

for author in author_venue:
    venues = author_venue[author]
    # print(author, venues)
    venues = set(venues)
    if 'none' in venues:
        venues.remove('none')
    venues = list(venues)
    random.shuffle(venues)
    num_samples = max(1, len(venues)//10)
    author_venue[author] = venues[:num_samples]
    # print(author, len(author_venue[author]))

vocab = set()

with open('../dataset/vocab.txt') as file:
    for line in file:
        vocab.add(line.strip())

with open('../dataset/edges_hc.txt', 'w') as hcfile:
    for author, venues in author_venue.items():
        for venue in set(venues):
            hcfile.write('@{} {} 1\n'.format(author, venue))

with open('../dataset/edges_hw.txt', 'w') as hwfile:
    for author, titles in author_title.items():
        for title in titles:
            words = set(title.split('-'))
            for word in words:
                if word in vocab:
                    hwfile.write('@{} {} 1\n'.format(author, word))
