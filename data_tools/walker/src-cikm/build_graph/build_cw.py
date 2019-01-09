#coding:utf-8

vocab_file_name = '../dataset/vocab.txt'
venues_file_name = '../dataset/venues.txt'
paper_venue_file_name = '../dataset/paper_title_venue.txt'

vocab = set()
venues = set()

with open(vocab_file_name) as file:
    for line in file:
        vocab.add(line.strip())

with open(venues_file_name) as file:
    for line in file:
        venues.add(line.strip())

paper_venue_cnt = {}

with open(paper_venue_file_name) as file:
    for line in file:
        paper_id, title, venue = line.strip().split()
        if venue not in venues:
            continue
        words = set(title.split('-'))
        for word in words:
            if word not in vocab:
                continue
            if word not in paper_venue_cnt:
                paper_venue_cnt[word] = {}
            if venue not in paper_venue_cnt[word]:
                paper_venue_cnt[word][venue] = 0

            paper_venue_cnt[word][venue] += 1

with open('../dataset/edges_cw.txt', 'w') as file:
    for word in paper_venue_cnt:
        for venue, cnt in paper_venue_cnt[word].items():
            file.write('{} {} {}\n'.format(word, venue, cnt))


