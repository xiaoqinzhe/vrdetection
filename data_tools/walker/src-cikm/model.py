#coding:utf-8
import multiprocessing as mp
import time

import numpy as np
import tensorflow as tf

class Model(object):

    def __init__(self, prob_mat, neg_ratio=5, embedding_dim=128, learning_rate=0.05, batch_size=200000, iterations=300, name_scope='default'):
        self.neg_ratio = neg_ratio
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iterations = iterations
        self.name_scope = name_scope

        self.prob_mat = prob_mat
        self.num_nodes = prob_mat.shape[0]
        self.weight_mat = self.prob_to_weight(prob_mat)
        self.indegree = np.sum(prob_mat, axis=0)
        self.indegree /= np.sum(self.indegree)

    def prob_to_weight(self, prob_mat):
        prob_max = np.max(prob_mat, axis=1)
        indices = (prob_mat>1e-4).nonzero()
        weight_mat = np.zeros(prob_mat.shape)
        weight_mat[indices[0],indices[1]] = np.max([prob_mat[indices[0],indices[1]], prob_mat[indices[1], indices[0]]], axis=0)/np.max([prob_max[indices[0]], prob_max[indices[1]]])
        return weight_mat
    
    def build_computational_graph(self):
        with tf.name_scope(self.name_scope):
            embeddings = tf.Variable(
                tf.random_uniform((self.num_nodes, self.embedding_dim), -1.0, 1.0),
                name="embeddings", 
            )
        
            u_pos = tf.placeholder(tf.int32, name='u_pos')
            v_pos = tf.placeholder(tf.int32, name='v_pos')
            w_pos = tf.placeholder(tf.float32, name='w_pos')
            u_neg = tf.placeholder(tf.int32, name='u_neg')
            v_neg = tf.placeholder(tf.int32, name='v_neg')
            w_neg = tf.placeholder(tf.float32, name='w_neg')

            emb_u_pos = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings, u_pos), 1, name='emb_u_pos')
            emb_v_pos = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings, v_pos), 1, name='emb_v_pos')
            emb_u_neg = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings, u_neg), 1, name='emb_u_neg')
            emb_v_neg = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings, v_neg), 1, name='emb_v_neg')

            dot_prod_pos = tf.reduce_sum(tf.multiply(emb_u_pos, emb_v_pos), 1, name='dot_prod_pos')
            dot_prod_neg = tf.reduce_sum(tf.multiply(emb_u_neg, emb_v_neg), 1, name='dot_prod_neg')

            loss_pos = -tf.reduce_sum(tf.multiply(w_pos, tf.log_sigmoid(5*dot_prod_pos)), name='loss_pos')
            loss_neg_1 = -tf.reduce_sum(tf.multiply(w_neg, tf.log_sigmoid(5*dot_prod_neg)), name='loss_neg_1')
            loss_neg_2 = -tf.reduce_sum(tf.multiply(1-w_neg, tf.log_sigmoid(-5*dot_prod_neg)), name='loss_neg_2')

            loss = loss_pos + loss_neg_1 + self.neg_ratio*loss_neg_2

        self.embeddings = embeddings
        self.placeholders = (u_pos, v_pos, w_pos, u_neg, v_neg, w_neg)
        self.loss = loss
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess

    def batches(self):
        num_nodes = self.num_nodes
        prob_mat = self.prob_mat
        batch_size = self.batch_size
        neg_ratio = self.neg_ratio
        indegree = self.indegree

        all_samp_pos = np.transpose((prob_mat>1e-4).nonzero())
        all_samp_neg = np.zeros(all_samp_pos.shape, dtype=np.int32)
        num_samples = all_samp_pos.shape[0]
        np.random.shuffle(all_samp_pos)
        all_samp_neg[:,0] = all_samp_pos[:,0]
        all_samp_neg[:,1] = np.random.choice(num_nodes, (num_samples,), p=indegree)
        
        start_index = 0
        end_index = min(start_index+batch_size, num_samples)
        while start_index<num_samples:
            samp_pos = all_samp_pos[start_index:end_index,:]
            samp_neg = all_samp_neg[start_index:end_index,:]
            yield samp_pos, samp_neg
            start_index = end_index
            end_index = min(start_index+batch_size, num_samples)

    def train_one_epoch(self):
        u_pos, v_pos, w_pos, u_neg, v_neg, w_neg = self.placeholders
        weight_mat = self.weight_mat

        sum_loss = 0.0
        for samp_pos, samp_neg in self.batches():

            samp_u_pos = samp_pos[:,0]
            samp_v_pos = samp_pos[:,1]
            samp_w_pos = weight_mat[samp_u_pos,samp_v_pos]
            samp_u_neg = samp_neg[:,0]
            samp_v_neg = samp_neg[:,1]
            samp_w_neg = weight_mat[samp_u_neg, samp_v_neg]

            _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
                u_pos: samp_u_pos,
                v_pos: samp_v_pos,
                w_pos: samp_w_pos,
                u_neg: samp_u_neg,
                v_neg: samp_v_neg,
                w_neg: samp_w_neg,
            })

            sum_loss += cur_loss

        return sum_loss

    def train(self):
        print('Training...')
        for i in range(self.iterations):
            epoch_loss = self.train_one_epoch()
            print('{}\tepoch {}/{}\t\tloss {}'.format(self.name_scope, i+1, self.iterations, epoch_loss))

    def get_embeddings(self):
        sess = self.sess
        embeddings = self.embeddings
        return np.array(sess.run(tf.nn.l2_normalize(embeddings.eval(session=sess), 1)))
