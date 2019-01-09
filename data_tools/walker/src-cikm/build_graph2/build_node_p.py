#coding:utf-8

with open('../dataset2/paper_selected.txt') as ifile, open('../dataset2/node_p.txt', 'w') as ofile:
    for line in ifile:
        paper, title, venue = line.strip().split()
        ofile.write('#{} paper\n'.format(paper))
