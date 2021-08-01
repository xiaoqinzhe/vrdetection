import numpy as np

import multiprocessing

class EvalRunnerMP:
    """
    A multi-processing runner for evaluation in testing
    """
    def __init__(self, task_func, task_capacity=10, data_capacity=100):
        self._task_func = task_func
        self.counter = 0
        self.processes = []
        self.task_capacity = task_capacity
        self.data_capacity = data_capacity

    def start_processes(self, n_processes=1):
        self.task_queue = multiprocessing.Queue(self.task_capacity)
        self.data_queue = multiprocessing.Queue(self.data_capacity)
        for n in range(n_processes):
            p = multiprocessing.Process(target=self._worker_main, args=(self.task_queue, self.data_queue))
            p.daemon = True
            p.start()
            self.processes.append(p)

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

    def _worker_main(self, task_queue, data_queue):
        """
        generate sample from task queue and put the sample
        into a data queue in the form of tf feed_dict
        """
        # from pympler import tracker, summary, muppy
        # import gc
        # memory_tracker = tracker.SummaryTracker()
        i = 0
        # print(gc.isenabled())
        # gc.set_debug(gc.DEBUG_STATS)
        while True:
            # if i % 10 == 0: memory_tracker.print_diff()

            # gc.collect()
            task = task_queue.get()
            # print("task get")
            res = self._task_func(task)
            # print("task comp", res)
            if res is None:
                continue
            data_queue.put(res)
            i += 1


def _compute_gt_target(pred_boxes, pred_class_scores, gt_boxes):
    """
    compute which gt gets mapped to each predicted box
    """

    num_boxes = pred_boxes.shape[0]
    # map predicted boxes to ground-truth
    gt_targets = np.zeros(num_boxes).astype(np.int32)
    gt_target_iou = np.zeros(num_boxes)
    gt_target_iou.fill(-1)

    for j in range(num_boxes):
        # prepare inputs
        bbox = pred_boxes[j].astype(float)
        # compute max IoU over classes
        # for c in xrange(1, num_classes):
        for c in range(pred_class_scores.shape[1]):
            bb = bbox[4*c:4*(c+1)]
            if gt_boxes.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(gt_boxes[:, 0], bb[0])
                iymin = np.maximum(gt_boxes[:, 1], bb[1])
                ixmax = np.minimum(gt_boxes[:, 2], bb[2])
                iymax = np.minimum(gt_boxes[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                        (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

                overlaps = inters / uni
                max_iou_class = np.max(overlaps)
                max_k_class = np.argmax(overlaps)

                # select max over classes
                if max_iou_class > gt_target_iou[j]:
                    gt_target_iou[j] = max_iou_class
                    gt_targets[j] = max_k_class

    return gt_targets, gt_target_iou


def ground_predictions(sg_entry, roidb_entry, ovthresh=0.5):
    """
    ground graph predictions onto ground truth annotations
    """

    # get predictions
    boxes = sg_entry['boxes']
    class_scores = sg_entry['scores']
    num_boxes = boxes.shape[0]

    # get ground-truth
    gt_inds = np.where(roidb_entry['max_overlaps'] == 1)[0]
    gt_boxes = roidb_entry['boxes'][gt_inds].astype(float)

    # compute which gt index each roi gets mapped to
    gt_targets, gt_target_iou = _compute_gt_target(boxes, class_scores, gt_boxes)

    # filter out predictions with low IoUs
    filter_inds = np.where(gt_target_iou > ovthresh)[0]

    # make sure each gt box is referenced only once
    # if referenced more than once, use the one that
    # has the maximum IoU
    gt_to_pred = {} # {gt_ind: pred_ind}
    for j in range(num_boxes):
        gti = gt_targets[j] # referenced gt ind
        if gti in gt_to_pred:
            pred_ind = gt_to_pred[gti]
            if gt_target_iou[j] > gt_target_iou[pred_ind]:
                gt_to_pred[gti] = j
        elif j in filter_inds: # also must survive filtering
            gt_to_pred[gti] = j

    return gt_to_pred
