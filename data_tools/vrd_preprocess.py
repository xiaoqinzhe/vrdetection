"""
preprocessing vrd dataset into npy
"""
import os, json, argparse
import cv2
import pprint
import numpy as np

def get_inds(file):
    strs = json.load(open(file))
    str_to_ind = {}
    ind_to_str = []
    for i, str in enumerate(strs):
        str_to_ind[str] = i
        ind_to_str.append(str)
    return str_to_ind, ind_to_str

def get_objs(objs, class_to_ind):
    boxes = []
    labels = []
    for obj in objs:
        if len(obj['names']) > 1: print(obj['names'])
        if obj['names'][0] in class_to_ind:
            labels.append(class_to_ind[obj['names'][0]])
        else:
            print("name %s is not in class list!! skipped..." % obj['names'][0])
            continue
        bbox = obj['bbox']
        boxes.append([bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']])
    return boxes, labels

def get_rels(rels, predicate_to_ind):
    relations = []
    for rel in rels:
        if rel['relationship'] not in predicate_to_ind:
            print("name %s is not in predicate list!! skipped..." % rel['relationship'])
            continue
        relations.append([rel['objects'][0], rel['objects'][1], predicate_to_ind[rel['relationship']]])
    return relations

def equal_objs(obj1, obj2):
    if obj1['category'] == obj2['category'] and obj1['bbox'] == obj2['bbox']:
        return True
    return False

def update_objs(obj, objs):
    for i, o in enumerate(objs):
        if equal_objs(o, obj):
            return i
    objs.append(obj)
    return len(objs) - 1

def get_objs_rels(rels):
    objs = []
    relations = []
    for rel in rels:
        sub_id = update_objs(rel['subject'], objs)
        obj_id = update_objs(rel['object'], objs)
        relations.append([sub_id, obj_id, rel['predicate']])
    boxes, labels = [], []
    for obj in objs:
        box = obj['bbox']
        boxes.append([box[2], box[0], box[3], box[1]])
        labels.append(obj['category'])
    return boxes, labels, relations

def preprocess(json_file, ims_path, save_file, class_to_ind, ind_to_class, predicate_to_ind, ind_to_predicate, exist_rels=None):
    is_train = 'train' in json_file
    info = json.load(open(json_file))
    save_info = []
    print("images amount = %d"%len(info))
    box_count, pred_count = 0, 0
    zs_count=0
    # recording rels that exists in training data
    if is_train:
        exist_rels = np.zeros([len(ind_to_class), len(ind_to_class), len(ind_to_predicate)], dtype=np.bool)
    for i, filename in enumerate(info):
        d = info[filename]
        if i % 1000 == 0:
            print("processing %d"%i)
        im_file = os.path.join(ims_path, filename)
        im = cv2.imread(im_file)
        if im is None:
            if os.path.exists(im_file):
                print("destroyed image: %s id: %d" % (im_file, i))
            else:
                print("missing image: %s id: %d. (you can try to change the img file from .gif to .jpg to fix this.)" % (im_file, i))
            continue
        height, width = im.shape[0], im.shape[1]
        boxes, labels, rels = get_objs_rels(d)
        if len(boxes) == 0:
            continue
        save_info.append({
            "image_filename" : ims_path.split("/")[-2]+"/"+filename,
            "image_width" : width,
            "image_height" : height,
            "boxes" : boxes,
            "labels" : labels,
            "relations" : rels,
        })
        if is_train:
            for rel in rels:
                exist_rels[rel[0], rel[1], rel[2]] = True
        else:
            zero_shot_tags = np.zeros(len(rels), np.bool)
            for i, rel in enumerate(rels):
                if not exist_rels[rel[0], rel[1], rel[2]]:
                    zero_shot_tags[i] = True
                    zs_count+=1
            save_info[-1]['zero_shot_tags'] = zero_shot_tags.tolist()
        box_count += len(boxes)
        pred_count += len(rels)
    final_save = {
        "data": save_info,
        "class_to_ind": class_to_ind,
        "ind_to_class": ind_to_class,
        "predicate_to_ind": predicate_to_ind,
        "ind_to_predicate": ind_to_predicate,
    }
    with open(save_file, 'w') as f:
        json.dump(final_save, f)
    print("saved images = %d" % len(save_info))                                 # 3780     954
    print("average boxes per image = %f" % ((box_count*1.0)/len(save_info)))    # 26430    6728      7
    print("saved relationships = %f" % ((pred_count*1.0)/len(save_info)))         # 30355    7632      8
    print("zero shot count = {}".format(zs_count))
    print("done.")
    return exist_rels

def check(json_file, name, ind_to_class, ind_to_predicate):
    info = json.load(open(json_file))
    save_info = []
    print("images amount = %d"%len(info))
    box_count, pred_count = 0, 0
    for i, filename in enumerate(info):
        d = info[filename]
        if i % 1000 == 0:
            print("processing %d"%i)
        if (filename.find(name) == -1):
            continue
        print(filename)
        boxes, labels, rels = get_objs_rels(d)
        print(boxes)
        for rel in rels:
            print(rel)
            print(ind_to_class[labels[rel[0]]], ind_to_predicate[rel[2]], ind_to_class[labels[rel[1]]])
        break

def check2(json_file, ims_path):
    info = json.load(open(json_file))
    print("images amount = %d"%len(info))
    icount = 0
    up_acc = 0.0
    im_count = 0
    for i, filename in enumerate(info):
        d = info[filename]
        if i % 1000 == 0:
            print("processing %d"%i)
        im_file = os.path.join(ims_path, filename)
        im = cv2.imread(im_file)
        if im is None:
            if os.path.exists(im_file):
                print("destroyed image: %s id: %d" % (im_file, i))
            else:
                print(
                "missing image: %s id: %d. (you can change the img file from .gif to .jpg to fix this.)" % (im_file, i))
            continue
        boxes, labels, rels = get_objs_rels(d)
        if len(boxes) == 0:
            continue
        rel_objs = [[rel[0],rel[1]] for rel in rels]
        dup_count = 0
        c = 0.0
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                cc = rel_objs.count([i, j])
                if cc > 0:
                    c += 1
                    if cc > 1:
                        dup_count += 1
        if dup_count >= 1:
            print("warning", filename, dup_count)
            icount += 1
        c /= len(rels)
        up_acc += c
        im_count += 1
        # print(boxes)
        # for rel in rels:
        #     print(rel)
        #     print(ind_to_class[labels[rel[0]]], ind_to_predicate[rel[2]], ind_to_class[labels[rel[1]]])
    print("total {}/{} are duplicated".format(icount, len(info)))
    print("max accuracy is {}/{}".format(up_acc/im_count, im_count))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="/hdd/datasets/vrd/vrd/", type=str)
    parser.add_argument('--save_path', default="./data/vrd/", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    class_to_ind, ind_to_class = get_inds(os.path.join(data_path, 'objects.json'))
    predicate_to_ind, ind_to_predicate = get_inds(os.path.join(data_path, 'predicates.json'))

    exist_rels = preprocess(os.path.join(data_path, 'annotations_train.json'), os.path.join(data_path, 'sg_train_images/'),
               os.path.join(save_path, "train.json",),
               class_to_ind, ind_to_class, predicate_to_ind, ind_to_predicate)
    preprocess(os.path.join(data_path, 'annotations_test.json'), os.path.join(data_path, 'sg_test_images/'),
               os.path.join(save_path, "test.json"),
               class_to_ind, ind_to_class, predicate_to_ind, ind_to_predicate, exist_rels=exist_rels)

    # check(os.path.join(data_path, 'annotations_test.json'), "8646018805_d914413321_b.jpg", ind_to_class, ind_to_predicate)
    #
    # check2(os.path.join(data_path, 'annotations_test.json'), os.path.join(data_path, 'sg_test_images/'))