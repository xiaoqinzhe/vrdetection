#coding:utf-8
authors1 = {}
authors2 = {}
with open('paper_labels.txt') as file:
    for line in file:
        author, paper, label = line.strip().split()
        if author not in authors1:
            authors1[author] = []
        authors1[author].append(line)

with open('author_paper_label.txt') as file:
    for line in file:
        author, paper, label = line.strip().split()
        if author not in authors2:
            authors2[author] = []
        authors2[author].append(line)

with open('author_paper_label_final.txt', 'w') as file:
    for author, papers in authors2.items():
        for line in papers:
            file.write(line)

    for author, papers in authors1.items():
        if author in authors2:
            continue
        for line in papers:
            file.write(line)