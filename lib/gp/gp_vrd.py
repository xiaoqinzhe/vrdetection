import sys
sys.path.insert(0, '.')

from gp.gp import GP
from datasets.vrd import vrd
import numpy as np
import copy
from datasets.sg_eval import eval_relation_recall, get_oo_id
# from scipy import stats

def get_db(dataset_name, image_set="train", num_im = -1):
    path = '../data/'
    if dataset_name.startswith("vg"):
        path += "vg/"
    return vrd(image_set, dataset_name, path, num_im=num_im)

def get_vrd_results(dataset_name, image_set):
    path = '../data/'
    if dataset_name.startswith("vg"):
        path += "vg/"
    path += "{}/vrd_results_{}_pred_cls.npy".format(dataset_name, image_set)
    return np.load(path, allow_pickle=True)

def get_prior(dataset_name, filename):
    path = '../data/' + dataset_name
    if dataset_name.startswith('vg'):
        path = '../data/vg/' + dataset_name
    filename = path + "/" + filename
    prior = np.load(filename, allow_pickle=True)
    print("load prior from {}".format(filename))
    return prior

def prepare_roidb(roidb_batch):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    for roidb in roidb_batch:
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb['max_classes'] = max_classes
        roidb['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        # nonzero_inds = np.where(max_overlaps > 0)[0]
        # assert all(max_classes[nonzero_inds] != 0)
    return roidb_batch

class GPVrd(GP):
    def __init__(self, dataset_name, eval_mode, top_k=1, num_terminals=3, popsize=100, h=10, t=20, gsize=2, gh=3, gt=4, num_variables=2,
                 max_iterator=1000, F=0.1, CR=0.5, multi_processors=0, task_capacity=100, batch_size=10, test_batch_size=10,
                 save_suffix="", training=False):
        super(GPVrd, self).__init__(popsize, num_terminals=num_terminals, h=h, t=t, gsize=gsize, gh=gh, gt=gt,
                                    num_variables=num_variables, max_iterator=max_iterator, F=F, CR=CR,
                                    multi_processors=multi_processors, task_capacity=task_capacity)
        if training:
            self.train_imdb = get_db(dataset_name, 'test')
            self.train_roidb = prepare_roidb(self.train_imdb.roidb)
            self.num_train_samples = self.train_imdb.num_images
        self.test_imdb = get_db(dataset_name, 'test')
        self.test_roidb = prepare_roidb(self.test_imdb.roidb)
        self.num_test_samples = self.test_imdb.num_images
        self.train_batch_size = batch_size
        self.test_batch_size = test_batch_size
        if training:
            self._shuffle_train_inds()
            self.get_next_train_index()
        self._shuffle_test_inds()
        self.get_next_test_index()
        if training:
            self.train_vrd_results = get_vrd_results(dataset_name, "test")
        self.test_vrd_results = get_vrd_results(dataset_name, "test")
        self.top_k = top_k
        self.eval_mode = eval_mode
        self.prior = get_prior(dataset_name, "lang_prior.npy")
        self.dataset_prior = get_prior(dataset_name, "dataset_gp.npy").item()
        self.test_all_fitness = None
        self.test_eval = True
        self.good_individuals = []
        self.good_fitness = []
        self.save_suffix = save_suffix
        self.data_path = '../data/{}/gp/'.format("vrd" if dataset_name == "vrd" else "vg/"+dataset_name)

    def _shuffle_train_inds(self):
        self._cur = 0
        self._cur_inds = []
        self.train_sample_index = np.random.permutation(np.arange(self.num_train_samples))

    def get_next_train_index(self):
        if self._cur + self.train_batch_size > self.num_train_samples:
            self._shuffle_train_inds()
        self._cur_inds = self.train_sample_index[self._cur: self._cur+self.train_batch_size]
        self._cur += self.train_batch_size
        return self._cur_inds

    def _shuffle_test_inds(self):
        self._cur_test = 0
        self._cur_test_inds = []
        self.test_sample_index = np.random.permutation(np.arange(self.num_test_samples))

    def get_next_test_index(self):
        if self._cur_test + self.test_batch_size > self.num_test_samples:
            self._shuffle_test_inds()
        self._cur_test_inds = self.test_sample_index[self._cur_test: self._cur_test+self.test_batch_size]
        self._cur_test += self.test_batch_size
        return self._cur_test_inds

    def on_iterator_start(self, iter):
        # self.get_next_train_index()
        pass

    def on_iterator_end(self, iter):
        self.get_next_train_index()
        if iter % 1 == 0:
            # self.test_eval = True
            # fitness, origin_fitness = self.test(self.population[self.best_index], self._cur_test_inds)
            # self.get_next_test_index()
            # print("test fitness: ", fitness, origin_fitness)
            self.test_all()
        # test all
        # if iter % 100 == 0:
        #     self.test_all()
        # pass

    def test_all(self, index=None, save=True):
        if index == None:
            index = self.best_index
        self.test_eval = True if self.test_all_fitness is None else False
        fitness, origin_fitness = self.test(self.population[index], np.arange(len(self.test_roidb)))
        self.test_all_fitness = origin_fitness
        if save and self.compare_fitness(fitness, origin_fitness):
            self.good_fitness.append(fitness)
            self.good_individuals.append(copy.deepcopy(self.population[index]))
            if save:
                np.save(self.data_path+"{}_good".format(self.eval_mode) + self.save_suffix, [self.good_individuals, self.good_fitness])
            # print(index, self.population[index])
            # print(self.good_individuals[-1])
        print("test all fitness: ", fitness, origin_fitness)
        print(index, self.population[index])
        self.test_eval = True

    def set_extra_data(self, i, individual):
        return [self._cur_inds]

    def test(self, individual, test_inds):
        data = [0, individual]
        if not self.use_multi_process:
            data.extend([[], test_inds])
            fitness, origin_fitness = self.cal_fitness(data, "test")[1]
        else:
            for test_i in test_inds:
                data = [0, individual, [], [test_i]]
                self.mp_runner.put_task(data)
            fitness = []
            origin_fitness = []
            for i in range(len(test_inds)):
                res = self.mp_runner.data_queue_get()
                fitness.append(res[1])
                origin_fitness.append(res[2])
            fitness = self.fitness_add(fitness)
            if self.test_eval:
                origin_fitness = self.fitness_add(origin_fitness)
                return fitness, origin_fitness
        return fitness, self.test_all_fitness

    def cal_fitness(self, data, mode="train"):
        index = data[0]
        individual = data[1]
        mode = "train" if len(data) == 3 else "test"
        if mode == "train":
            samples_inds = data[2]
            vrd_results = self.train_vrd_results
            roidb = self.train_roidb
            imdb = self.train_imdb
        else:
            samples_inds = data[3]
            vrd_results = self.test_vrd_results
            roidb = self.test_roidb
            imdb = self.test_imdb
        # print(index, samples_inds)
        root, sub_roots = self.decode(individual)
        # self.print_node(root)

        result_dict = {}
        result_dict[self.eval_mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
        origin_result_dict = {}
        origin_result_dict[self.eval_mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
        for i in samples_inds:
            sg_entry = vrd_results[i]
            predicate_preds = np.copy(sg_entry['relations'])
            num_predicates = len(predicate_preds[0][0])
            rel_weights = sg_entry['rel_weights']
            fg_prob_sub = sg_entry['fg_prob_sub']
            fg_prob_obj = sg_entry['fg_prob_obj']

            # # 3
            rel_weights = np.tile(rel_weights[:, np.newaxis], num_predicates)
            fg_prob_sub = np.tile(fg_prob_sub[:, np.newaxis], num_predicates)
            fg_prob_obj = np.tile(fg_prob_obj[:, np.newaxis], num_predicates)
            sub_obj_prior = np.zeros_like(rel_weights, dtype=np.float32)

            # k = 0
            # classes = sg_entry['cls_preds']
            # for a in range(len(predicate_preds[0])):
            #     for b in range(len(predicate_preds[0])):
            #         if a == b: continue
            #         sub_obj_prior[k, 1:] = self.prior[get_oo_id(classes[a], classes[b], imdb.num_classes)]
            #         k += 1
            #
            # # 4
            # prediction = np.zeros_like(rel_weights, dtype=np.float32)
            # k = 0
            # for a in range(len(predicate_preds[0])):
            #     for b in range(len(predicate_preds[0])):
            #         if a == b: continue
            #         prediction[k] = predicate_preds[a][b]
            #         k += 1
            #
            # # 5,6,7
            # o2p_dsprior = np.zeros_like(rel_weights, dtype=np.float32)
            # o2o_dsprior = np.zeros_like(rel_weights, dtype=np.float32)
            # s_dsprior = np.zeros_like(rel_weights, dtype=np.float32)
            # o_dsprior = np.zeros_like(rel_weights, dtype=np.float32)
            # k = 0
            # for a in range(len(predicate_preds[0])):
            #     for b in range(len(predicate_preds[0])):
            #         if a == b: continue
            #         # print(o2p_dsprior.shape)
            #         # print(self.dataset_prior, classes)
            #         o2p_dsprior[k, 1:] = self.dataset_prior['o2p_m'][classes[a], classes[b]]
            #         o2o_dsprior[k, :] = self.dataset_prior['o2o_m'][classes[a], classes[b]]
            #         s_dsprior[k, :] = self.dataset_prior['o_m'][classes[a]]
            #         o_dsprior[k, :] = self.dataset_prior['o_m'][classes[b]]
            #         k += 1

            classes = roidb[i]['gt_classes']
            for a, rel in enumerate(roidb[i]["gt_relations"]):
                sub_obj_prior[a, 1:] = self.prior[get_oo_id(classes[rel[0]], classes[rel[1]], imdb.num_classes)]

            # 4
            prediction = np.zeros_like(rel_weights, dtype=np.float32)
            k = 0
            for a, rel in enumerate(roidb[i]["gt_relations"]):
                prediction[a] = predicate_preds[rel[0]][rel[1]]

            # 5,6,7
            o2p_dsprior = np.zeros_like(rel_weights, dtype=np.float32)
            o2o_dsprior = np.zeros_like(rel_weights, dtype=np.float32)
            s_dsprior = np.zeros_like(rel_weights, dtype=np.float32)
            o_dsprior = np.zeros_like(rel_weights, dtype=np.float32)
            k = 0
            for a, rel in enumerate(roidb[i]["gt_relations"]):
                o2p_dsprior[a, 1:] = self.dataset_prior['o2p_m'][classes[rel[0]], classes[rel[1]]]
                o2o_dsprior[a, :] = self.dataset_prior['o2o_m'][classes[rel[0]], classes[rel[1]]]
                s_dsprior[a, :] = self.dataset_prior['o_m'][classes[rel[0]]]
                o_dsprior[a, :] = self.dataset_prior['o_m'][classes[rel[1]]]

            noise = np.random.rand(rel_weights.shape[0], rel_weights.shape[1])
            self.terminals[0] = prediction
            self.terminals[1] = sub_obj_prior
            self.terminals[2] = o2p_dsprior
            # self.terminals[3] = 2

            # noise = np.random.rand(rel_weights.shape[0], rel_weights.shape[1])
            # self.terminals[0] = rel_weights
            # self.terminals[1] = fg_prob_sub
            # self.terminals[2] = fg_prob_obj +noise
            # self.terminals[3] = sub_obj_prior
            # self.terminals[4] = prediction
            # self.terminals[5] = np.max([o2p_dsprior, sub_obj_prior], 0)
            # self.terminals[6] = o2o_dsprior
            # self.terminals[7] = s_dsprior
            # self.terminals[8] = o_dsprior

            # print(np.shape(rel_weights), np.shape(fg_prob_sub))

            values = self.cal_tree(root, sub_roots)
            # dh3
            # values = prediction*(sub_obj_prior+o2p_dsprior)*(rel_weights+o2o_dsprior)*\
            #          (fg_prob_sub+s_dsprior)*(fg_prob_obj+o_dsprior)/16
            # values = prediction*np.max([sub_obj_prior, o2p_dsprior], 0)*\
            #          np.max([rel_weights, o2o_dsprior], 0)*\
            #          np.max([fg_prob_sub, s_dsprior], 0)*\
            #          np.max([fg_prob_obj, o_dsprior], 0)
            # values = rel_weights*fg_prob_obj*fg_prob_sub
            # values = prediction * np.sqrt(o2p_dsprior) * np.min([o2p_dsprior,o2o_dsprior],0)

            # k = 0
            # for a in range(len(predicate_preds[0])):
            #     for b in range(len(predicate_preds[0])):
            #         if a == b: continue
            #         predicate_preds[a][b] = values[k]
            #         k += 1

            for a, rel in enumerate(roidb[i]["gt_relations"]):
                predicate_preds[rel[0]][rel[1]] = values[a]

            sg_entry_t = {}
            for key in sg_entry:
                sg_entry_t[key] = sg_entry[key]
            sg_entry_t["relations"] = predicate_preds
            sg_entry_t["num_predicates"] = sg_entry["num_predicates"] = imdb.num_predicates
            sg_entry_t["num_classes"] = sg_entry["num_classes"] = imdb.num_classes

            eval_relation_recall(sg_entry_t, roidb[i], result_dict,
                                 self.eval_mode,
                                 0.5,
                                 top_k=self.top_k, use_prediction=True,
                                 use_prior=False, use_weight=False, use_fg_weight=False, prior=self.prior
                                 )
            # if mode == "test" and self.test_eval:
            #     eval_relation_recall(sg_entry, roidb[i], origin_result_dict,
            #                          self.eval_mode,
            #                          0.5,
            #                          top_k=self.top_k, use_prediction=True,
            #                          use_prior=True, use_weight=True, use_fg_weight=True, prior=self.prior
            #                          )
        # print(result_dict)
        fitness = self.define_fitness(result_dict)
        if mode == "test":
            origin_fitness = self.define_fitness(origin_result_dict)
            # if fitness > origin_fitness:
            #     print(index, fitness, origin_fitness)
            return [index, fitness, origin_fitness]
        return [index, fitness]

    def define_fitness(self, result_dict):
        return [np.mean(result_dict[self.eval_mode + '_recall'][100]),
                  np.mean(result_dict[self.eval_mode + '_recall'][50])]

    def compare_fitness(self, fitness1, fitness2):
        for i in range(len(fitness1)):
            if fitness1[i] < fitness2[i]: return False
        return True

    def fitness_add(self, fitness):
        return np.mean(fitness, axis=0)

    def print_node(self, node):
        if node is None: return
        print(node.value)
        for child in node.children:
            self.print_node(child)

if __name__ == '__main__':
    dataset_name = "vrd"  # vg/vg_msdn
    top_k = 70
    eval_mode = "pred_cls"
    training = False
    gp = GPVrd(dataset_name, eval_mode=eval_mode, top_k=top_k, save_suffix="_top{}_1".format(top_k),
               num_terminals=3, h=10, t=20,
               batch_size=10, test_batch_size=954, popsize=100,
               max_iterator=5000, multi_processors=20, task_capacity=40001,
               training=training)
    gp.start_processor()
    #
    # gp.evolute(random_seed=666, path=gp.data_path+"{}/".format(top_k), save_freq=10)


    # gp.load("../data/vrd/gp/iter_1000.npy")
    # for i in range(gp.popsize):
    #     gp.test_all(i, save=False)
    # print(gp.population[gp.best_index])
    # gp.load("../data/vrd/gp/iter_500.npy")
    # print(gp.population)

    # if dataset_name.startswith("vg"): dataset_name = "vg/"+dataset_name
    # good_inds = np.load("../data/{}/gp/good_top{}_1.npy".format(dataset_name, top_k), allow_pickle=True)
    # great_id = np.argmax(good_inds[1])
    # good = good_inds[0][great_id]
    # print(good_inds[1][great_id], good)
    # gp.population[0] = good

    # dh
    gp.population[0] = [2, 100, 2, 102, 0, 102, 7, 102, 100, 8, 101, 101, 102, 100, 100, 101, 100, 101, 101, 102, 102, 102, 100, 102, 102, 102, 102, 101, 100, 101, 1000, 5, 6, 1001, 1001, 1001, 1000, 1000, 2, 1001, 1001, 1001, 1001, 1000]
        # [6, 7, 8, 7, 101, 6, 2, 102, 100, 3, 101, 102, 100, 100, 100, 100, 101, 101, 102, 102, 102, 100, 100, 101, 101, 101, 101, 100, 100, 102, 1, 1001, 2, 1000, 1000, 1000, 1001, 1, 0, 2, 1000, 1000, 1000, 1001]
        # [2, 2, 5, 100, 9, 2, 3, 8, 0, 4, 105, 104, 101, 106, 107, 110, 102, 106, 102, 101, 109, 110, 105, 103, 107, 104, 106, 109, 102, 107, 1000, 3, 1, 1000, 1001, 1000, 1000, 5, 1000, 1001, 1000, 1000, 1000, 1000]
    gp.population[1] = [2, 100, 3, 0, 103, 101, 102, 100, 100, 100, 100, 100, 100, 100, 101, 101, 100, 102, 100, 101, 101, 100, 101, 102, 100, 101, 100, 102, 101, 100, 1001, 1001, 7, 1000, 1000, 1001, 1000, 3, 5, 2, 1000, 1000, 1000, 1000]
    gp.population[2] = [2, 100, 7, 101, 102, 100, 100, 100, 100, 100, 100, 100, 100, 100, 101, 101, 100, 102, 100, 101, 101, 100, 101, 102, 100, 101, 100, 102, 101, 100, 1001, 1001, 7, 1000, 1000, 1001, 1000, 3, 5, 2, 1000, 1000, 1000, 1000]
    gp.population[3] = [2, 100, 7, 101, 102, 100, 100, 100, 100, 100, 100, 100, 100, 100, 101, 101, 100, 102, 100, 101, 101, 100, 101, 102, 100, 101, 100, 102, 101, 100, 1001, 1001, 7, 1000, 1000, 1001, 1000, 3, 5, 2, 1000, 1000, 1000, 1000]
    gp.population[4] = [2, 100, 7, 101, 102, 100, 100, 100, 100, 100, 100, 100, 100, 100, 101, 101, 100, 102, 100, 101, 101, 100, 101, 102, 100, 101, 100, 102, 101, 100, 1001, 1001, 7, 1000, 1000, 1001, 1000, 3, 5, 2, 1000, 1000, 1000, 1000]
    gp.test_all(0, save=False)

#     stats.ttest_rel()
# # test net, detection file
