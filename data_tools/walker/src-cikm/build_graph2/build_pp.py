#coding:utf-8

paper_authors = {}
with open('../dataset2/selected_paper_authors.txt') as file:
    for line in file:
        sp = line.strip().split()
        paper = sp[0]
        authors = set(sp[1:])
        paper_authors[paper] = authors

paper_authors = list(paper_authors.items())

with open('../dataset2/edges_pp.txt', 'w') as file:
    for i, (paper1, p1authors) in enumerate(paper_authors):
        for paper2, p2authors in paper_authors[i+1:]:
            num_coauthors = len(p1authors&p2authors)
            if num_coauthors>1:
                file.write('#{} #{} {}\n'.format(paper1, paper2, num_coauthors-1))
