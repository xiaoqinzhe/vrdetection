#coding:utf-8

papers = {}
vocab = set()
with open('../dataset/vocab.txt') as file:
    for line in file:
        vocab.add(line.strip())

with open('../dataset/paper_selected.txt') as ifile, open('../dataset/edges_pc.txt', 'w') as pcfile, open('../dataset/edges_pw.txt', 'w') as pwfile:
    for line in ifile:
        paper, title, venue = line.strip().split()
        if venue!='none':
            pcfile.write('#{} {} 1\n'.format(paper, venue))
        words = set(title.split('-'))
        for word in words:
            if word in vocab:
                pwfile.write('#{} {} 1\n'.format(paper, word))
