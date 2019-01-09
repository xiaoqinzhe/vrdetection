#coding:utf-8

papers = set()
with open('../dataset2/author_paper_label.txt') as file:
    for line in file:
        author, paper, label = line.strip().split()
        papers.add(paper)

with open('../dataset2/paper_title_venue.txt') as ifile, open('../dataset/paper_selected.txt', 'w') as ofile:
    for line in ifile:
        paper, title, venue = line.strip().split()
        if paper in papers:
            ofile.write(line)
