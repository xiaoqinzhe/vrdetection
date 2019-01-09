#coding:utf-8
import os
import multiprocessing as mp

import numpy as np
import networkx as nx

from pprint import pprint

from util import HeteroGraphLookup, load_graph, random_file_name

class Walker(object):

    def __init__(self, graph, num_draws=100):
        self.graph = graph
        self.num_draws = num_draws
        self.init_types_and_nodes()
        self.init_next_node_table()

    def init_types_and_nodes(self):
        print('Initializing graph node types...')
        self.node_type = {node_label:node_attr['node_type'] for node_label, node_attr in self.graph.nodes(data=True)}
        self.all_types = set([typ for label, typ in self.node_type.items()])
        self.type_nodes = {typ:set() for typ in self.all_types}
        for node, typ in self.node_type.items():
            self.type_nodes[typ].add(node)

    def init_next_node_table(self):
        print('Building sampling table...')
        self.next_node = {node:{'nodes':None, 'seq':None, 'probs':None, 'index':None} for node in self.graph.nodes()}

        for cur_node in self.graph.nodes():
            neighbors = set(self.graph.neighbors(cur_node))
            all_neighbors = []
            all_probs = []
            for typ in self.all_types:
                typ_neighbors = list(neighbors&self.type_nodes[typ])
                probs = np.array([self.graph[cur_node][neighbor]['weight'] for neighbor in typ_neighbors])
                probs = probs/np.sum(probs)
                all_neighbors += typ_neighbors
                all_probs = np.concatenate([all_probs, probs])
            all_probs = all_probs/np.sum(all_probs)

            self.next_node[cur_node]['nodes'] = all_neighbors
            self.next_node[cur_node]['probs'] = all_probs


    def draw_next_node(self, cur_node):
        table = self.next_node[cur_node]
        if len(table['nodes'])==0:
            return None
        if table['index']==self.num_draws or table['index'] is None:
            table['seq'] = np.random.choice(table['nodes'], size=self.num_draws, p=table['probs'])
            table['index'] = 0
        index = table['index']
        table['index'] += 1
        return table['seq'][index]

    def walk_one_round(self, sequences, walk_length):
        this_round_seq = []
        for node in self.graph.nodes():
            cur_node = node
            seq = [cur_node]
            for step in range(walk_length-1):
                next_node = self.draw_next_node(cur_node)
                if next_node is None:
                    break
                seq.append(next_node)
                cur_node = next_node
            if len(seq)==walk_length:
                this_round_seq.append(seq)
        sequences += this_round_seq

    def walk(self, num_walks=10, walk_length=80, workers=8):
        print('Walking...')
        with mp.Manager() as manager:
            sequences = manager.list()
            pool = mp.Pool(workers)
            for walk in range(num_walks):
                pool.apply_async(self.walk_one_round, args=(sequences, walk_length,))
            pool.close()
            pool.join()
            all_sequences = list(sequences)
        return all_sequences

    def walk_score(self, start_node, dest_node, walk_length=10, num_walks=1000):
        score = 0
        for i_walk in range(num_walks):
            cur_node = start_node
            for i_pos in range(walk_length):
                next_node = self.draw_next_node(cur_node)
                if next_node is None:
                    break
                if next_node==dest_node:
                    score += walk_length-i_pos
                    break
                cur_node = next_node
        return score/num_walks


class TransProbCalculator(object):

    def __init__(self, graph, lookup):
        self.lookup = lookup
        self.graph = graph

    def proc_cal_prob_mat(self, node_type, num_nodes, sequences, window_size):
        result = np.zeros((num_nodes, num_nodes), dtype=np.float)
        for seq in sequences:
            for i, node_u in enumerate(seq):
                if self.lookup.node_index_to_type(node_u)!=node_type:
                    continue
                for node_v in seq[i+1:i+window_size]:
                    if self.lookup.node_index_to_type(node_v)!=node_type:
                        continue
                    u = self.lookup.node_global_index_to_type_index(node_u)
                    v = self.lookup.node_global_index_to_type_index(node_v)
                    result[u, v] += 1
        # tmp_file = random_file_name('/hdd/zjj/tmp/nparray', 'npy')
        # np.save(tmp_file, result)
        # return tmp_file
        return result

    def cal_prob_mat(self, node_type, sequences, window_size=10, workers=8):
        print('Calculating prob matrix...')

        num_nodes = len(self.lookup.type2labels[node_type])
        mat = np.zeros((num_nodes, num_nodes), dtype=np.float)
        # load_per_proc = len(sequences)//8
        mat = self.proc_cal_prob_mat(node_type, num_nodes, sequences, window_size)
        # results = []
        # pool = mp.Pool(workers)
        # for i in range(workers):
        #     if i==workers-1:
        #         results.append(pool.apply_async(
        #             self.proc_cal_prob_mat,
        #             args=(
        #                 node_type,
        #                 num_nodes,
        #                 sequences[i*load_per_proc:],
        #                 window_size,
        #             ),
        #         ))
        #     else:
        #         results.append(pool.apply_async(
        #             self.proc_cal_prob_mat,
        #             args=(
        #                 node_type,
        #                 num_nodes,
        #                 sequences[i*load_per_proc:(i+1)*load_per_proc],
        #                 window_size
        #             ),
        #         ))
        # pool.close()
        # pool.join()
        # for mat_file in results:
        #     result = np.load(mat_file.get())
        #     mat += result


        norm_mat = np.apply_along_axis(lambda row: row/np.sum(row), axis=1, arr=mat)
        norm_mat[np.isnan(norm_mat)] = 0

        # for mat_file in results:
        #     os.system('rm {}'.format(mat_file.get()))
        return norm_mat
