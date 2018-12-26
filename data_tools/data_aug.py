import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

import numpy as np
from utils.cython_bbox import bbox_overlaps
import json

def get_aug_boxes(gt_box, width, height, size, iou_thresh=0.5):
    scale_range = 0.3
    dis_range = 0.3
    aug_boxes = np.zeros([0, 4], np.int)
    while(size>0):
        scales, dis = [], []
        for i in range(2*size):
            r = np.random.rand()+0.5
            while(r<1-scale_range or r>1.0+scale_range): r = np.random.rand()+0.5
            r2 = np.random.rand() + 0.5
            while (r2 < 1 - scale_range or r2 > 1.0 + scale_range): r2 = np.random.rand() + 0.5
            scales.append([r, r2])
        scales = np.array(scales) - 1
        # print(scales)
        for i in range(2*size):
            r = 2*np.random.rand() - 1.0
            while(r<-dis_range or r>dis_range): r = np.random.rand() - 1.0
            r2 = 2 * np.random.rand() - 1.0
            while (r2 < -dis_range or r2 > dis_range): r2 = np.random.rand() - 1.0
            dis.append([r, r2])
        dis = np.array(dis)
        # print(dis)
        # scale box
        boxes = np.tile([gt_box], [len(dis), 1])
        w = gt_box[2] - gt_box[0]
        h = gt_box[3] - gt_box[1]
        boxes[:, 0] = np.floor(gt_box[0] - w * scales[:, 0] / 2)
        boxes[:, 1] = np.floor(gt_box[1] - h * scales[:, 1] / 2)
        boxes[:, 2] = np.ceil(gt_box[2] + w * scales[:, 0] / 2)
        boxes[:, 3] = np.ceil(gt_box[3] + h * scales[:, 1] / 2)
        # move boxes
        boxes[:, 0] = np.floor(boxes[:, 0] + w * dis[:, 0])
        boxes[:, 1] = np.floor(boxes[:, 1] + h * dis[:, 1])
        boxes[:, 2] = np.ceil(boxes[:, 2] + w * dis[:, 0])
        boxes[:, 3] = np.ceil(boxes[:, 3] + h * dis[:, 1])
        # checking
        # print(scales)
        # print(dis)
        # print(gt_box)
        # print(boxes)
        boxes = clip_boxes(boxes, width, height)
        boxes = match_iou(boxes, gt_box, iou_thresh)
        # add to
        if boxes.shape[0]>size:
            inds = np.random.permutation(boxes.shape[0])
            boxes = boxes[inds[:size]]
        aug_boxes = np.vstack((aug_boxes, boxes))
        size = size - len(boxes)
    return aug_boxes

def match_iou(boxes, gt_box, iou_thresh=0.5):
    overlaps = bbox_overlaps(gt_box[np.newaxis,].astype(np.float), boxes.astype(np.float))[0]
    inds = np.where(overlaps>iou_thresh)
    return boxes[inds]

def clip_boxes(boxes, width, height):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    # y1 >= 0
    boxes[:, 0:2] = np.maximum(boxes[:, 0:2], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], width)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], height)
    return boxes

def aug_vrd_roidb(roidb):
    for im in range(len(roidb)):
        if im%100==0: print("process %dth image..."%im)
        data = roidb[im]
        width, height = data['image_width'], data['image_height']
        gt_boxes = np.array(data['boxes'])
        gt_labels = np.array(data['labels'])
        gt_relations = np.array(data['relations'])
        aug_boxes = np.zeros([0, 4], np.int)
        aug_labels = np.zeros([0], np.int)
        aug_relations = np.zeros([0, 3], np.int)
        gt_to_auginds = [[] for _ in range(len(gt_labels))]
        for i, gt_box in enumerate(gt_boxes):
            boxes = get_aug_boxes(gt_box, width, height, size=4)
            boxes = np.vstack((gt_box, boxes))
            gt_to_auginds[i].extend(np.arange(len(aug_boxes), len(aug_boxes)+len(boxes)))
            # print(gt_to_auginds[i])
            aug_boxes = np.vstack((aug_boxes, boxes))
            labels = np.tile([i], [len(boxes)])
            aug_labels = np.hstack((aug_labels, labels))
        for i in range(gt_relations.shape[0]):
            gt_rel = gt_relations[i]
            rels = []
            max_rels_per_gtrel = -1
            count=0
            # print(gt_rel[0], gt_rel[1])
            # print(gt_to_auginds[gt_rel[0]])
            # print(gt_to_auginds[gt_rel[1]])
            for rel1 in gt_to_auginds[gt_rel[0]]:
                for rel2 in gt_to_auginds[gt_rel[1]]:
                    # print(rel1, rel2)
                    # assert rel1!=rel2
                    rels.append([rel1, rel2, gt_rel[2]])
                    count += 1
            rels = np.array(rels)
            if max_rels_per_gtrel>0:
                rels = rels[:max_rels_per_gtrel]
            aug_relations = np.vstack((aug_relations, rels))
        roidb[im]['boxes'] = aug_boxes
        roidb[im]['labels'] = aug_labels
        roidb[im]['relations'] = aug_relations

def aug_vrd():
    json_file = './data/vrd/train.json'
    save_file = './data/vrd/train_aug.pickle'
    info = json.load(open(json_file))
    print("augmenting...")
    aug_vrd_roidb(info['data'])
    import pickle
    pickle.dump(info, open(save_file,'w'))
    print('save to {}. done'.format(save_file))

if __name__ == "__main__":
    aug_vrd()