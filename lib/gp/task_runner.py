import multiprocessing

class TaskRunner:
    def __init__(self, task_func, task_generator=None, task_capacity=10, data_capacity=10):
        self._task_func = task_func
        self.processes = []
        self.task_capacity = task_capacity
        self.data_capacity = data_capacity
        self.task_queue = multiprocessing.Queue(self.task_capacity)
        self.data_queue = multiprocessing.Queue(self.data_capacity)
        self.task_generator = task_generator

    def start_processes(self, n_processes=1):
        # t = multiprocessing.Process(target=self._asyn_put_task, args=(self.task_queue))
        # t.start()
        # self.processes.append(t)
        for n in range(n_processes):
            p = multiprocessing.Process(target=self._worker_main, args=(self.task_queue, self.data_queue))
            p.daemon = True
            p.start()
            self.processes.append(p)

    def data_queue_get(self, block=True, timeout=None):
        return self.data_queue.get(block, timeout)

    def get_data(self):
        size = self.data_queue.qsize()
        # print(size)
        result = []
        for i in range(size):
            result.append(self.data_queue.get())
        # print("end", result)
        return result

    def put_task(self, data):
        self.task_queue.put(data)

    def _asyn_put_task(self, task_queue):
        for t in self.task_generator:
            task_queue.put(t)

    def _worker_main(self, task_queue, data_queue):
        while True:
            task = task_queue.get()
            res = self._task_func(task)
            if res is None:
                continue
            data_queue.put(res)
