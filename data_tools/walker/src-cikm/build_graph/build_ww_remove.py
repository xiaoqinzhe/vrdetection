#coding:utf-8

vocab_file_name = '../dataset/vocab_add.txt'
ww_add_file_name = '../dataset/edges_ww_add.txt'

vocab = set()
degree = {}
num_edges = 0
with open(vocab_file_name) as file:
    for line in file:
        num_edges += 1
        vocab.add(line.strip())

with open(ww_add_file_name) as file:
    for line in file:
        num_edges += 1
        w1, w2, cnt = line.strip().split()
        if w1 not in degree:
            degree[w1] = 0
        if w2 not in degree:
            degree[w2] = 0
        degree[w1] += 1
        degree[w2] += 1

words = list(degree.items())
words.sort(key=lambda elem: elem[1], reverse=True)

remove_words = set()
for word, deg in words:
    if deg<3000:
        break
    remove_words.add(word)

remove_words_l = list(remove_words)
remove_words_l.sort()
with open('../dataset/vocab_remove.txt', 'w') as file:
    for word in remove_words_l:
        file.write('{}\n'.format(word))

num_edges = 0

with open(ww_add_file_name) as ifile, open('../dataset/edges_ww_final.txt', 'w') as ofile:
    for line in ifile:
        w1, w2, cnt = line.strip().split()
        if w1 in remove_words or w2 in remove_words:
            continue
        ofile.write('{} {} {}\n'.format(w1, w2, cnt))
        num_edges += 1

print(num_edges)
