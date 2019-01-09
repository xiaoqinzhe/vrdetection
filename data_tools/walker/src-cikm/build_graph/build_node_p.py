#coding:utf-8

with open('../dataset/paper_selected.txt') as ifile, open('../dataset/node_p.txt', 'w') as ofile:
    for line in ifile:
        paper, title, venue = line.strip().split()
        ofile.write('#{} paper\n'.format(paper))
