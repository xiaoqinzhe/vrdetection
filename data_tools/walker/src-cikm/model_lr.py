#coding:utf-8
import multiprocessing as mp
import time

import numpy as np
import tensorflow as tf

class Model(object):

    def __init__(self, prob_mat, embedding_dim=128, learning_rate=0.5, batch_size=10, iterations=50, name_scope='default'):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iterations = iterations
        self.name_scope = name_scope

        self.prob_mat = prob_mat
        self.num_nodes = prob_mat.shape[0]
        self.indegree = np.sum(prob_mat, axis=0)
        self.node_prob = self.indegree
        self.node_prob = self.node_prob/np.sum(self.node_prob)
    
    def build_computational_graph(self):
        with tf.name_scope(self.name_scope):
            embeddings = tf.Variable(
                tf.random_uniform((self.num_nodes, self.embedding_dim), -1.0, 1.0),
                name="embeddings", 
            )
        
            u_src = tf.placeholder(tf.int32, name='u_src')
            v_pos = tf.placeholder(tf.int32, name='v_pos')
            v_neg = tf.placeholder(tf.int32, name='v_neg')
            w_pos = tf.placeholder(tf.float32, name='w_pos')
            w_neg = tf.placeholder(tf.float32, name='w_neg')
            sign = tf.placeholder(tf.float32, name='sign')

            emb_u_src = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings, u_src), 1, name='emb_u_src')
            emb_v_pos = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings, v_pos), 1, name='emb_v_pos')
            emb_v_neg = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings, v_neg), 1, name='emb_v_neg')

            dot_pos = tf.reduce_sum(tf.multiply(emb_u_src, emb_v_pos), 1, name='dot_pos')
            dot_neg = tf.reduce_sum(tf.multiply(emb_u_src, emb_v_neg), 1, name='dot_neg')

            loss_triplet = -tf.reduce_sum(
                tf.multiply(
                    tf.multiply(sign, w_pos-w_neg),
                    tf.log_sigmoid(tf.multiply(sign, dot_pos-dot_neg)),
                )
            )

            loss = loss_triplet

        self.embeddings = embeddings
        self.placeholders = (u_src, v_pos, v_neg, w_pos, w_neg, sign)
        self.loss = loss
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess

    def batches(self):
        num_nodes = self.num_nodes
        prob_mat = self.prob_mat
        batch_size = self.batch_size
        node_prob = self.node_prob

        all_samp_pos = np.transpose((prob_mat>1e-4).nonzero())
        num_samples = all_samp_pos.shape[0]

        triplets = np.zeros((num_samples, 3), dtype=np.int32)
        triplets[:,0:2] = all_samp_pos
        triplets[:,2] = np.random.choice(num_nodes, (num_samples,), p=node_prob)
        np.random.shuffle(triplets)
        
        start_index = 0
        end_index = min(start_index+batch_size, num_samples)
        while start_index<num_samples:
            yield triplets[start_index:end_index,:]
            start_index = end_index
            end_index = min(start_index+batch_size, num_samples)

    def train_one_epoch(self):
        u_src, v_pos, v_neg, w_pos, w_neg, sign = self.placeholders
        prob_mat = self.prob_mat

        sum_loss = 0.0
        for samp_triplets in self.batches():
            samp_u_src = samp_triplets[:,0]
            samp_v_pos = samp_triplets[:,1]
            samp_v_neg = samp_triplets[:,2]
            samp_w_pos = prob_mat[samp_u_src, samp_v_pos]
            samp_w_neg = prob_mat[samp_u_src, samp_v_neg]
            samp_sign = np.ones((samp_w_pos.shape[0],))
            samp_sign[(samp_w_pos-samp_w_neg)<=0] = -1

            _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
                u_src: samp_u_src,
                v_pos: samp_v_pos,
                v_neg: samp_v_neg,
                w_pos: samp_w_pos,
                w_neg: samp_w_neg,
                sign:  samp_sign,
            })

            sum_loss += cur_loss

        return sum_loss

    def train(self):
        if not hasattr(self, 'sess'):
            self.build_computational_graph()
        print('Training...')
        for i in range(self.iterations):
            epoch_loss = self.train_one_epoch()
            print('{}\tepoch {}/{}\t\tloss {}'.format(self.name_scope, i+1, self.iterations, epoch_loss))
        return self

    def get_embeddings(self):
        sess = self.sess
        embeddings = self.embeddings
        return np.array(sess.run(tf.nn.l2_normalize(embeddings.eval(session=sess), 1)))
