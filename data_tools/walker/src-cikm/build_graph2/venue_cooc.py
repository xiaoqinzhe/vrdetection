#coding:utf-8

paper_venue_file = '../dataset2/paper_title_venue.txt'
citation_file = '../dataset2/citation.txt'
citation_venue_file = '../dataset2/citation_venue.txt'

paper_venue = {}

with open(paper_venue_file) as file:
    for line in file:
        paper_id, title, venue = line.strip().split()
        if venue=='none':
            continue
        paper_venue[paper_id] = venue

with open(citation_file) as ifile, open(citation_venue_file, 'w') as ofile:
    for line in ifile:
        papers = line.strip().split()
        venues = []
        for paper in papers:
            if paper in paper_venue:
                venues.append(paper_venue[paper])
        if len(venues)>1:
            ofile.write('{}\n'.format(' '.join(venues)))
