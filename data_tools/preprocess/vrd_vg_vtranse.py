import json, os


def inter_names(names1, names2):
    inames = []
    ids1 = []
    ids2 = []
    for i, name in enumerate(names1):
        if name not in names2: continue
        ind = names2.index(name)
        inames.append(name)
        ids1.append(i)
        ids2.append(ind)
    return inames, ids1, ids2

def check_data(data, keep_cls_ids, keep_pred_ids):
    cls_id2ex = {id: False for id in keep_cls_ids}
    pred_id2ex = {id: False for id in keep_pred_ids}
    for d in data:
        labels = d['labels']
        for rel in d['relations']:
            if labels[rel[0]] in cls_id2ex and labels[rel[1]] in cls_id2ex and rel[2] in pred_id2ex:
                cls_id2ex[labels[rel[0]]] = True
                cls_id2ex[labels[rel[1]]] = True
                pred_id2ex[rel[2]] = True
    zero_cls_inds, zero_pred_inds = [], []
    for i, id in enumerate(cls_id2ex):
        if not cls_id2ex[id]: zero_cls_inds.append(i)
    for i, id in enumerate(pred_id2ex):
        if not pred_id2ex[id]: zero_pred_inds.append(i)
    return zero_cls_inds, zero_pred_inds

def get_id_names(names):
    return names, {name: i for i, name in enumerate(names)}

def filter_data(data, cls_id2ind, pred_id2ind):
    ndata = []
    print(len(data))
    num_rels = 0
    num_objs = 0
    for d in data:
        nd = {
            'image_filename': d['image_filename'],
            "image_width": d['image_width'],
            "image_height": d['image_height']
        }
        keep_obj_ids = set()
        keep_rel_ids = set()
        labels = d['labels']
        for i, rel in enumerate(d['relations']):
            if labels[rel[0]] in cls_id2ind and labels[rel[1]] in cls_id2ind and rel[2] in pred_id2ind:
                keep_obj_ids.add(rel[0])
                keep_obj_ids.add(rel[1])
                keep_rel_ids.add(i)
        if len(keep_obj_ids) == 0 or len(keep_rel_ids) == 0:
            continue
        obj_id2ind = {id: ind for ind, id in enumerate(keep_obj_ids)}
        boxes, labels, relations = [], [], []
        for id in keep_obj_ids:
            boxes.append(d['boxes'][id])
            labels.append(cls_id2ind[d['labels'][id]])
        for i, rel in enumerate(d['relations']):
            if i in keep_rel_ids:
                relations.append([obj_id2ind[rel[0]], obj_id2ind[rel[1]], pred_id2ind[rel[2]]])
        nd['boxes'] = boxes
        nd['labels'] = labels
        nd['relations'] = relations
        nd['zero_shot_tags'] = []
        ndata.append(nd)
        num_objs += len(boxes)
        num_rels += len(relations)
    print("imgs: {}".format(len(ndata)), "objs: {} {}".format(num_objs, num_objs/len(ndata)), "rels: {} {}".format(num_rels, num_rels/len(ndata)))
    return ndata

def vrd_vg_inter(vrd_path, vg_path):
    vrd_train = json.load(open(vrd_path+'train.json'))
    vrd_test = json.load(open(vrd_path+'test.json'))
    vg_train = json.load(open(vg_path + 'train.json'))
    vg_test = json.load(open(vg_path + 'test.json'))
    # vrd = {
    #     "labels": vrd_train['labels'],
    #     "relations": vrd_train['relations'],
    # }
    keep_cls_names, vrd_keep_cls_ids, vg_keep_cls_ids = inter_names(vrd_train['ind_to_class'], vg_train['ind_to_class'])
    keep_pred_names, vrd_keep_pred_ids, vg_keep_pred_ids = inter_names(vrd_train['ind_to_predicate'], vg_train['ind_to_predicate'])
    print("keep classes {}, ".format(len(keep_cls_names)), keep_cls_names)
    print("keep predicates {}, ".format(len(keep_pred_names)), keep_pred_names)

    zero_cls_inds, zero_pred_inds = check_data(vrd_train['data'], vrd_keep_cls_ids, vrd_keep_pred_ids)
    if not len(zero_cls_inds) == 0 or not len(zero_pred_inds) == 0:
        print("warning: some classes/predicates have no samples")
        return
    zero_cls_inds, zero_pred_inds = check_data(vg_train['data'], vg_keep_cls_ids, vg_keep_pred_ids)
    if not len(zero_cls_inds) == 0 or not len(zero_pred_inds) == 0:
        print("warning: some classes/predicates have no samples")
        return
    print("cleaning...")

    vrd_cls_id2ind = {id: i for i, id in enumerate(vrd_keep_cls_ids)}
    vrd_pred_id2ind = {id: i for i, id in enumerate(vrd_keep_pred_ids)}
    vg_cls_id2ind = {id: i for i, id in enumerate(vg_keep_cls_ids)}
    vg_pred_id2ind = {id: i for i, id in enumerate(vg_keep_pred_ids)}
    id2class, class2id = get_id_names(keep_cls_names)
    id2pred, pred2id = get_id_names(keep_pred_names)

    vrd_train['data'] = filter_data(vrd_train['data'], vrd_cls_id2ind, vrd_pred_id2ind)
    vrd_test['data'] = filter_data(vrd_test['data'], vrd_cls_id2ind, vrd_pred_id2ind)
    vg_train['data'] = filter_data(vg_train['data'], vg_cls_id2ind, vg_pred_id2ind)
    vg_test['data'] = filter_data(vg_test['data'], vg_cls_id2ind, vg_pred_id2ind)
    for dataset in [vrd_train, vrd_test, vg_train, vg_test]:
        dataset["ind_to_class"] = id2class
        dataset["class_to_ind"] = class2id
        dataset["ind_to_predicate"] = id2pred
        dataset["predicate_to_ind"] = pred2id

    # save
    new_vrd_path = "./data/tl_vrd/"
    new_vg_path = "./data/tl_vg/"
    if not os.path.exists(new_vrd_path):  os.makedirs(new_vrd_path)
    if not os.path.exists(new_vg_path):  os.makedirs(new_vg_path)
    json.dump(vrd_train, open(new_vrd_path+"train.json", 'w'))
    json.dump(vrd_test, open(new_vrd_path + "test.json", 'w'))
    json.dump(vg_train, open(new_vg_path + "train.json", 'w'))
    json.dump(vg_test, open(new_vg_path + "test.json", 'w'))

    print("save successfully.")

if __name__ == '__main__':
    data_path = './data/'
    dataset_name = 'vrd_vg'
    vrd_path = data_path + 'vrd/'
    vg_path = data_path + 'vg/vg_vtranse/'

    vrd_vg_inter(vrd_path, vg_path)

# result:
# keep classes 68,  ['person', 'sky', 'building', 'truck', 'bus', 'table', 'shirt', 'chair', 'car', 'train', 'tree', 'boat', 'hat', 'grass', 'road', 'motorcycle', 'jacket', 'wheel', 'umbrella', 'plate', 'bike', 'clock', 'bag', 'shoe', 'laptop', 'desk', 'cabinet', 'counter', 'bench', 'tower', 'bottle', 'helmet', 'stove', 'lamp', 'bed', 'dog', 'mountain', 'horse', 'plane', 'roof', 'skateboard', 'bush', 'phone', 'sink', 'shelf', 'box', 'hand', 'cat', 'bowl', 'pillow', 'pizza', 'basket', 'elephant', 'kite', 'keyboard', 'plant', 'vase', 'pot', 'surfboard', 'paper', 'ball', 'bear', 'giraffe', 'tie', 'hydrant', 'engine', 'watch', 'suitcase']
# keep predicates 39,  ['on', 'wear', 'next to', 'above', 'behind', 'stand behind', 'under', 'near', 'walk', 'in', 'below', 'beside', 'over', 'hold', 'by', 'beneath', 'with', 'sit on', 'ride', 'carry', 'stand on', 'use', 'at', 'attach to', 'cover', 'touch', 'watch', 'against', 'inside', 'across', 'contain', 'drive on', 'eat', 'pull', 'lean on', 'fly', 'face', 'rest on', 'hit']
# cleaning...
# 3780
# imgs: 3487 objs: 17417 4.9948379696013765 rels: 16970 4.86664754803556
# 954
# imgs: 884 objs: 4420 5.0 rels: 4222 4.776018099547511
# 73794
# imgs: 26422 objs: 99599 3.7695481038528498 rels: 59046 2.234728635228219
# 25858
# imgs: 9254 objs: 35573 3.8440674303004108 rels: 21197 2.2905770477631293