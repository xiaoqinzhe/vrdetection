#coding:utf-8
import json

file_path = '/mnt/hdd2/dblp/dblp_ref.json'
citation_file_path = '/mnt/hdd2/cikm/citation.txt'
with open(file_path) as ifile, open(citation_file_path, 'w') as ofile:
    for line in ifile:
        paper = json.loads(line)
        if 'references' not in paper:
            continue
        output_papers = [paper['_id']]
        output_papers += paper['references']
        ofile.write('{}\n'.format(' '.join(output_papers)))

    