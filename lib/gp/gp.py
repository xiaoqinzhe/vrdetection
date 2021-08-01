import sys
sys.path.insert(0, '.')

import math, random, copy
from gp.task_runner import TaskRunner
import numpy as np
from utils.timer import Timer

import warnings
warnings.filterwarnings('ignore')

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

class GP:
    def __init__(self, popsize, num_terminals, h=20, t=10, gsize=2, gh=3, gt=4, num_variables=2,
                 max_iterator=1000, F=0.1, CR=0.5, multi_processors=0, task_capacity=100):
        self.h = h
        self.t = t
        # assert h<t, 't must larger than h'
        self.gsize = gsize
        self.gh = gh
        self.gt = gt
        self.popsize = popsize
        self.length = self.h + self.t + self.gsize*(self.gh + self.gt)
        self.num_terminals = num_terminals
        self.num_variables = num_variables
        self.population = [[] for _ in range(self.popsize)]
        self.fitness = [0.0 for _ in range(self.popsize)]
        self.terminals = [0 for _ in range(self.num_terminals)]
        self.variables = [0 for _ in range(self.num_variables)]
        self.max_iterator = max_iterator
        self.best_index = 0
        self.F = F
        self.CR = CR
        # functions setting
        self._init_func()

        # gene type
        self._gene_type = [0 for _ in range(self.h)]
        self._gene_type.extend([1 for _ in range(self.t)])
        for i in range(self.gsize):
            self._gene_type.extend([2 for _ in range(self.gh)])
            self._gene_type.extend([3 for _ in range(self.gt)])

        self._terminal_label_begin = 100
        self._variable_label_begin = 1000

        # multi process setting
        self._init_multiprocess(multi_processors, task_capacity)

    def _init_func(self):
        self.base_func_names = ["+", "-", "*", "/", "log", "sqrt", "min", "max"]
        self.num_child_of_func = [2, 2, 2, 2, 1, 1, 2, 2]  # + - * / log sqrt min max
        self.num_base_funcs = len(self.num_child_of_func)
        self.num_funcs = self.num_base_funcs + self.gsize

        self.func_freqs, self.func_probs  = [], []
        self.terminal_freqs, self.terminal_probs = [], []
        self.func_rate = 1.0

    def _init_multiprocess(self, multi_processors, task_capacity):
        self.task_capacity = task_capacity
        if multi_processors <= 0:
            self.use_multi_process = False
        else:
            self.use_multi_process = True
        self.multi_processors = multi_processors
        if self.use_multi_process:
            self.task_func = self.cal_fitness
            self.mp_runner = TaskRunner(self.task_func, task_capacity=self.task_capacity, data_capacity=self.popsize)

    def start_processor(self):
        self.mp_runner.start_processes(self.multi_processors)

    def base_func(self, func_label, data):
        if func_label == 0:
            return data[0] + data[1]
        if func_label == 1:
            return data[0] - data[1]
        if func_label == 2:
            return data[0] * data[1]
        if func_label == 3:
            # nonzero_inds = np.where(data[1]<1e-6)
            # res = np.zeros_like(data[0])
            # res[nonzero_inds] = data[0][nonzero_inds] / data[1][nonzero_inds]
            res = data[0] / data[1]
            np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            return res
            # if data[1] > 1e-6: return data[0]/data[1]
            # else: return np.zeros_like(data[0])
        if func_label == 4:
            res = np.log(data[0])
            np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            return res
        if func_label == 5:
            res = np.sqrt(data[0])
            np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            return res
        if func_label == 6:
            return np.min([data[0], data[1]], axis=0)
        if func_label == 7:
            return np.max([data[0], data[1]], axis=0)
        print("unsupported func label {}".format(func_label))
        exit(1)

    def decode(self, individual):
        root = self._decode_func(individual[:self.h+self.t], self.h, self.t)
        sub_roots = []
        for i in range(self.gsize):
            start = self.h+self.t+i*(self.gh+self.gt)
            end = start + self.gh + self.gt
            s = self._decode_func(individual[start:end], self.gh, self.gt)
            sub_roots.append(s)
        return root, sub_roots

    def _decode_func(self, l, h, t):
        root = Node(l[0])
        i, j = 0, 1
        nodes = [root]
        while i<j and i<h:
            num_child = self.num_child_of_func[l[i]] if l[i] < self.num_base_funcs else self.num_variables
            if l[i] >= self._terminal_label_begin and l[i] < self._variable_label_begin:
                num_child = 0
            for k in range(num_child):
                child = Node(l[j])
                nodes[i].children.append(child)
                nodes.append(child)
                j += 1
            i += 1
        return root

    def cal_tree(self, root, sub_roots=None):
        if root.value < self._terminal_label_begin:
            if len(root.children) <= 0: return 0.0  # wrong!!!
            if root.value < self.num_base_funcs:
                num_child = self.num_child_of_func[root.value]
            else: num_child = self.num_variables
            data = []
            for i in range(num_child):
                d = self.cal_tree(root.children[i], sub_roots)
                data.append(d)
            if root.value < self.num_base_funcs:
                return self.base_func(root.value, data)
            else:
                for i in range(len(data)):
                    self.variables[i] = data[i]
                return self.cal_tree(sub_roots[root.value-self.num_base_funcs], None)
        if root.value < self._variable_label_begin:
            return self.terminals[root.value-self._terminal_label_begin]
        return self.variables[root.value-self._variable_label_begin]

    def init_population(self):
        def rand_funcs(n, adf=True, base_func=True):
            start = 0
            num_func = self.num_base_funcs
            if adf: num_func += self.gsize
            if not base_func: start = self.num_base_funcs
            return [random.randint(start, num_func-1) for _ in range(n)]
        def rand_terminals(n):
            return [random.randint(self._terminal_label_begin, self._terminal_label_begin+self.num_terminals-1)
                    for _ in range(n)]
        def rand_variables(n):
            return [random.randint(self._variable_label_begin, self._variable_label_begin + self.num_variables-1)
                    for _ in range(n)]
        for i in range(self.popsize):
            for k in range(self.h):
                r = random.random()
                if r < 2/3.0:
                    self.population[i].extend(rand_funcs(1, adf=True))
                # if r < 1.0/3.0:
                #     self.population[i].extend(rand_funcs(1, adf=False))
                # elif r < 2.0/3.0:
                #     self.population[i].extend(rand_funcs(1, adf=True, base_func=False))
                else:
                    self.population[i].extend(rand_terminals(1))
            self.population[i].extend(rand_terminals(self.t))
            for j in range(self.gsize):
                for k in range(self.gh):
                    if random.random() < 1.0 / 2.0:
                        self.population[i].extend(rand_funcs(1, adf=False))
                    else:
                        self.population[i].extend(rand_variables(1))
                self.population[i].extend(rand_variables(self.gt))

    '''
       must to override
       data: list
    '''
    def cal_fitness(self, data):
        index = data[0]
        individual = data[1]
        root, sub_roots = self.decode(individual)
        fitness = self.cal_tree(root, sub_roots)
        return [index, fitness]

    # optional overrides
    def compare_fitness(self, fitness1, fitness2):
        return fitness1 < fitness2

    def on_iterator_start(self, iter):
        pass

    def on_iterator_end(self, iter):
        pass

    def set_extra_data(self, i, individual):
        return []

    def evaluate_population(self, population):
        fitness = [0 for _ in range(self.popsize)]
        for i in range(self.popsize):
            data = [i, population[i]]
            data.extend(self.set_extra_data(i, population[i]))
            if not self.use_multi_process:
                res = self.cal_fitness(data)
                fitness[res[0]] = res[1]
            else:
                self.mp_runner.put_task(data)
        if self.use_multi_process:
            for i in range(self.popsize):
                res = self.mp_runner.data_queue_get()
                fitness[res[0]] = res[1]
        return fitness

    def update_probs(self):
        a = 1.0
        num_funcs = self.num_base_funcs + self.gsize
        self.func_freqs, self.func_probs = [a for _ in range(num_funcs)], [0.0 for _ in range(num_funcs)]
        self.terminal_freqs, self.terminal_probs = \
            [a for _ in range(self.num_terminals)], [0.0 for _ in range(self.num_terminals)]
        func_count = 0.0
        for i in range(self.popsize):
            for j in range(self.h):
                value = self.population[i][j]
                if value < self._terminal_label_begin:
                    self.func_freqs[value] += 1
                    func_count += 1.0
                else: self.terminal_freqs[value-self._terminal_label_begin] += 1
            for j in range(self.h, self.h+self.t):
                self.terminal_freqs[self.population[i][j] - self._terminal_label_begin] += 1
        self.func_rate = func_count / (self.h*self.popsize)
        fn = func_count + num_funcs*a
        tn = (self.h+self.t)*self.popsize - func_count + self.num_terminals*a
        self.func_probs[0] = self.func_freqs[0] / fn
        for i in range(1, num_funcs):
            self.func_probs[i] = self.func_freqs[i]/fn + self.func_probs[i-1]
        self.terminal_probs[0] = self.terminal_freqs[0] / tn
        for i in range(1, self.num_terminals):
            self.terminal_probs[i] = self.terminal_freqs[i]/tn + self.terminal_probs[i-1]

    def roulette_wheel(self, prob):
        r = random.random()
        for i in range(len(prob)):
            if r < prob[i]:
                return i
        return len(prob)-1

    def get_best_index(self, fitness):
        best_i = 0
        for i in range(1, len(fitness)):
            if self.compare_fitness(fitness[i], fitness[best_i]):
                best_i = i
        return best_i

    def evolute(self, early_stopping=True, save_freq=1000, path="../data/", random_seed=666, verbose=True):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.init_population()
        # self.fitness = self.evaluate_population(self.population)
        new_population = copy.deepcopy(self.population)
        self.best_index = 0
        it = 0
        iter_timer = Timer()
        while it < self.max_iterator:
            iter_timer.tic()
            self.on_iterator_start(it)
            self.fitness = self.evaluate_population(self.population)
            self.best_index = self.get_best_index(self.fitness)
            if it==0: self.update_probs()
            # print(self.func_rate, self.func_freqs, self.terminal_freqs)
            for i in range(self.popsize):
                F = random.random()
                CR = random.random()
                r1 = random.randint(0, self.popsize-1)
                while r1 == i: r1 = random.randint(0, self.popsize-1)
                r2 = random.randint(0, self.popsize-1)
                while r2 == i or r2 == r1: r2 = random.randint(0, self.popsize - 1)
                k = random.randint(0, self.length-1)
                for j in range(self.length):
                    a, b = 0, 0
                    if self.population[i][j] != self.population[self.best_index][j]:
                        a = 1
                    if self.population[r1][j] != self.population[r2][j]:
                        b = 1
                    rate = 1 - (1-F*a)*(1-F*b)
                    if (random.random() < CR or j == k) and random.random() < rate:
                        if self._gene_type[j] == 0:
                            if random.random() < self.func_rate:
                                new_value = self.roulette_wheel(self.func_probs)
                            else: new_value = self.roulette_wheel(self.terminal_probs) + self._terminal_label_begin
                        elif self._gene_type[j] == 1:
                            new_value = self.roulette_wheel(self.terminal_probs) + self._terminal_label_begin
                        elif self._gene_type[j] == 2:
                            if random.random() < 0.5:
                                new_value = random.randint(0, self.num_base_funcs-1)
                            else:
                                new_value = random.randint(0, self.num_variables - 1) + self._variable_label_begin
                        else:
                            new_value = random.randint(0, self.num_variables-1) + self._variable_label_begin
                        new_population[i][j] = new_value
                    else:
                        new_population[i][j] = self.population[i][j]
            new_fitness = self.evaluate_population(new_population)
            for i in range(self.popsize):
                if self.compare_fitness(new_fitness[i], self.fitness[i]):
                    if self.compare_fitness(new_fitness[i], self.fitness[self.best_index]):
                        self.best_index = i
                    p = self.population[i]
                    self.population[i] = new_population[i]
                    new_population[i] = p
                    self.fitness[i] = new_fitness[i]
            it += 1
            iter_timer.toc()
            if verbose:
                print("iter {}, best fitness: {}, {}".format
                      (it, self.fitness[self.best_index], iter_timer.average_time))
            if it % save_freq == 0:
                filename = path + "/iter_{}".format(it)
                self.save(filename)
                print("save model to {}.".format(filename))
            self.on_iterator_end(it)
        # self.save(path+"/iter_{}".format(it))
        # test ?

    def save(self, filename):
        np.save(filename, [self.population, self.best_index])

    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.population = data[0]
        self.best_index = data[1]

if __name__ == "__main__":
    gp = GP(100, 3, h=3, t=4, gsize=1, gh=1, gt=2, multi_processors=10)
    gp.terminals = [1,2,3]
    root, sub_roots = gp.decode([0, 8, 1, 100, 101, 102, 100, 3, 1001, 1000])
    value = gp.cal_tree(root, sub_roots)
    print(value)
    gp.init_population()
    print(gp.population[66:68])
    gp.evolute()
