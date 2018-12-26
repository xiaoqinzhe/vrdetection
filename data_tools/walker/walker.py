# coding:utf-8

import tempfile
import multiprocessing as mp
import os
import sys
import time
from collections import namedtuple

import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize

def sliding_window(seq, window_size):
    for i in range(len(seq)-window_size+1):
        yield seq[i:i+window_size]

def sliding_window_lol(lol, window_size):
    for seq in lol:
        for i in range(len(seq)-window_size+1):
            yield seq[i:i+window_size]

class NextNodeManager(object):
    NODE_CACHE_SIZE = 100

    def __init__(self, neighbors, probs):
        self.neighbors = neighbors
        self.probs = probs
        self.seq = None
        self.curr = None

    def get_next_node(self):
        if self.curr is None or self.curr == self.NODE_CACHE_SIZE:
            self.seq = np.random.choice(self.neighbors, size=self.NODE_CACHE_SIZE, p=self.probs)
            self.curr = 0
        next_node = self.seq[self.curr]
        self.curr += 1
        return next_node


class Walker(object):
    @staticmethod
    def _get_num_workers(mode):
        if isinstance(mode, int):
            return mode

        mode_to_process_count = {
            'double': int(2 * mp.cpu_count()),
            'all': mp.cpu_count(),
            'half': mp.cpu_count() // 2,
            'single': 1,
        }

        if mode in mode_to_process_count:
            return mode_to_process_count[mode]

        raise Exception('Invalid multiprocess mode.')

    @staticmethod
    def _split_tasks(num_workers, num_tasks):
        ratio_per_worker = 1. / num_workers
        indices = [0]
        for i in range(1, num_workers + 1):
            indices.append(round(num_tasks * ratio_per_worker * i))
        return list(sliding_window(indices, 2))

    def __init__(self, num_walks=10, walk_length=80, weighted=False, multi_process='double'):
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.weighted = weighted
        self.multi_process = multi_process

    def build_sampling_table(self, graph, weighted):
        sampling_table = {}
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))

            if len(neighbors) == 0:
                neighbors = [node]
                probs = None
            else:
                if weighted:
                    probs = np.array([graph[node][neighbor]['weight'] for neighbor in neighbors])
                    probs = probs / np.sum(probs)
                else:
                    probs = None

            sampling_table[node] = NextNodeManager(neighbors, probs)
        return sampling_table

    def _process_walk_func(self, sampling_table, task_start, task_end, num_walks, walk_length):
        # np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        nodes = list(range(task_start, task_end))
        sequences = []
        for node in range(task_start, task_end):
            for w in range(num_walks):
                cur_node = node
                seq = [cur_node]
                for step in range(walk_length - 1):
                    next_node = sampling_table[cur_node].get_next_node()
                    if next_node is None:
                        break
                    seq.append(next_node)
                    cur_node = next_node
                if len(seq) == walk_length:
                    sequences.append(seq)
        return sequences

    def walk(self, graph):
        sampling_table = self.build_sampling_table(graph, self.weighted)

        num_workers = Walker._get_num_workers(self.multi_process)

        if num_workers == 1:
            return self._process_walk_func(
                sampling_table,
                task_start=0,
                task_end=graph.number_of_nodes(),
                num_walks=self.num_walks,
                walk_length=self.walk_length
            )

        tasks = Walker._split_tasks(num_workers, graph.number_of_nodes())
        pool = mp.Pool(num_workers)
        results = [
            pool.apply_async(
                self._process_walk_func,
                kwds={
                    'sampling_table': sampling_table,
                    'task_start': start,
                    'task_end': end,
                    'num_walks': self.num_walks,
                    'walk_length': self.walk_length,
                }
            )
            for start, end in tasks
        ]
        pool.close()
        pool.join()
        sequences = []
        for result in results:
            sequences.extend(result.get())
        return sequences