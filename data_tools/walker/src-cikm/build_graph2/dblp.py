#coding:utf-8

import re
import json

re_non_letter_digit_space = re.compile(r'[^A-Za-z0-9 ]+')
re_continuous_space = re.compile(r' +')
file_path = '../dataset2/dblp_ref.json'

def process_title(title):
    if title is None:
        return 'none'
    t = re_non_letter_digit_space.sub('', title)
    t = re_continuous_space.sub(' ', t)
    t = t.strip()
    t = t.lower()
    if len(t)==0:
        return 'none'
    return t

def process_venue(venue):
    if venue is None:
        return 'none'
    v = re_continuous_space.sub(' ', venue)
    v = v.strip()
    v = v.lower()
    if len(v)==0:
        return 'none'
    return '$'+v

with open(file_path) as file:
    for line in file:
        paper = json.loads(line)
        print('{} {} {}'.format(
            paper['_id'],
            '-'.join(process_title(paper.get('title', None)).split()),
            '-'.join(process_venue(paper.get('venue', None)).split()),
        ))
