#coding:utf-8

citation_venue_file = '../dataset/citation_venue.txt'

edges_cc = {}

with open(citation_venue_file) as file:
    for line in file:
        venues = line.strip().split()
        venues_s = list(set(venues))
        for i, v1 in enumerate(venues_s):
            for v2 in venues_s[i+1:]:
                if (v1,v2) not in edges_cc:
                    edges_cc[(v1,v2)] = 0
                    edges_cc[(v2,v1)] = 0
                cnt = min((venues.count(v1), venues.count(v2)))
                edges_cc[(v1,v2)] += 1
                edges_cc[(v2,v1)] += 1

printed = set()

with open('../dataset/edges_cc.txt', 'w') as file:
    for (v1,v2), cnt in edges_cc.items():
        if (v1,v2) in printed or (v2,v1) in printed:
            continue
        printed.add((v1,v2))
        printed.add((v2,v1))
        file.write('{} {} {}\n'.format(v1, v2, cnt))
