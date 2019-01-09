#coding:utf-8

file_name = '../dataset2/paper_title_venue.txt'

venues = set()
word_df = {}

with open(file_name) as file:
    for line in file:
        paper_id, title, venue = line.strip().split()
        words = title.split('-')
        for word in words:
            if word not in word_df:
                word_df[word] = set()
            word_df[word].add(venue)
        venues.add(venue)

venues.remove('none')

for word, venue in word_df.items():
    if 'none' in venue:
        venue.remove('none')

venues = list(venues)
venues.sort()

with open('../dataset2/venues.txt', 'w') as file:
    for venue in venues:
        file.write('{}\n'.format(venue))

words = list(word_df.keys())
words.sort()
with open('../dataset2/word_df.txt', 'w') as file:
    for word in words:
        if len(word)==1 or len(word_df[word])<3:
            continue
        df = len(word_df[word])/len(venues)
        file.write('{} {:.4f}\n'.format(word, df))
