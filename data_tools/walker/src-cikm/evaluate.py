#coding:utf-8
import sys
from pprint import pprint
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

score_file = sys.argv[1]

authors = {}

ranges = [(1,5), (6,10), (11, 20), (21, 50), (51, 100000)]

author_range_scores = {rang:[] for rang in ranges}

with open(score_file) as file:
    for line in file:
        author, paper, label, score = line.strip().split()
        label = int(label)
        score = float(score)

        if author not in authors:
            authors[author] = {'labels':[], 'scores':[]}
        
        authors[author]['labels'].append(label)
        authors[author]['scores'].append(score)

for author, arr in authors.items():
    arr['labels'] = np.array(arr['labels'])
    arr['scores'] = np.array(arr['scores'])

score_sum = 0
for author, arr in authors.items():
    y_true = arr['labels']
    scores = arr['scores']
    try:
        auc_score = roc_auc_score(y_true, scores)
    except:
        auc_score = 1.0

    max_f1_score = 0
    for score in scores:
        y_pred = np.zeros((len(scores),), dtype=np.int32)
        y_pred[scores>(score-1e-3)] = 1
        f1 = f1_score(y_true, y_pred, average='macro')
        max_f1_score = np.max([max_f1_score, f1])
    for lo, up in ranges:
        if lo<=(y_true==1).nonzero()[0].shape[0]<=up:
            author_range_scores[(lo,up)].append(auc_score)
    print(author, max_f1_score)
    score_sum += auc_score

print(score_sum/len(authors))
for (lo, up), scores in author_range_scores.items():
    print(lo, up, np.average(scores))
