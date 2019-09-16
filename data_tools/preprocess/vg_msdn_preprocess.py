"""
preprocessing vrd dataset into npy
"""
import os, json, argparse
import cv2
import pprint
import numpy as np

classes2ind = {}
predicates2ind = {}

def update_pred2ind(pred_name):
    if pred_name in predicates2ind:
        return predicates2ind[pred_name]
    predicates2ind[pred_name] = len(predicates2ind)
    return len(predicates2ind) - 1

def update_cls2ind(cls_name):
    if cls_name in classes2ind:
        return classes2ind[cls_name]
    classes2ind[cls_name] = len(classes2ind)
    return len(classes2ind) - 1

def get_objs_rels(data):
    boxes, labels = [], []
    bad_boxes = []
    for i, obj in enumerate(data['objects']):
        if obj['box'][2] == obj['box'][0] or obj['box'][3] == obj['box'][1]:
            bad_boxes.append(i)
            continue
            # print(obj['box'])
        obj_cls_ind = update_cls2ind(obj['class'])
        boxes.append(obj['box'])
        labels.append(obj_cls_ind)
    relations = []
    for rel in data['relationships']:
        pred_ind = update_pred2ind(rel['predicate'])
        sub_id, obj_id = rel['sub_id'], rel['obj_id']
        if [sub_id, obj_id, pred_ind] not in relations: relations.append([sub_id, obj_id, pred_ind])
    if len(bad_boxes) != 0:
        print('bad bounding box! correcting...')
        if len(bad_boxes) != 1: print('can not support!!!')
        nrels = []
        for i in range(len(bad_boxes)):
            st = bad_boxes[i]
            et = len(data['objects'])
            # print(bad_boxes, st, et)
            # print(relations)
            if i!=len(bad_boxes)-1: et = bad_boxes[i+1]
            for j in range(len(relations)):
                sid, oid, pid = relations[j]
                if sid == st or oid == st:
                    continue
                if sid > st and sid < et:
                    sid -= 1
                if oid > st and oid < et:
                    oid -= 1
                nrels.append([sid, oid, pid])
            relations = nrels
            # print(relations)

    return boxes, labels, relations

def preprocess(json_file, ims_path, save_file, exist_rels=None, num_classes=150, num_predicates=50):
    global predicates2ind, classes2ind
    is_train = 'train' in json_file
    info = json.load(open(json_file))
    #print(info[0])
    save_info = []
    print("images amount = %d"%len(info))
    box_count, pred_count = 0, 0
    zs_count=0
    # recording rels that exists in training data
    if is_train:
        exist_rels = np.zeros([num_classes, num_classes, num_predicates], dtype=np.bool)
    for i in range(len(info)):
        d = info[i]
        if i % 1000 == 0:
            print("processing %d"%i)
        filename = d['path']
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

        # print(d)
        # print(predicates2ind)
        # print(classes2ind)
        # print(boxes)
        # print(labels)
        # print(rels)
        #exit()
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
                    zs_count += 1
            save_info[-1]['zero_shot_tags'] = zero_shot_tags.tolist()
        box_count += len(boxes)
        pred_count += len(rels)
    ind2classes = ['' for _ in range(num_classes)]
    ind2predicates = ['' for _ in range(num_predicates)]
    for cls, ind in classes2ind.items():
        ind2classes[ind] = cls
    for pred, ind in predicates2ind.items():
        ind2predicates[ind] = pred
    final_save = {
        "data": save_info,
        "class_to_ind": classes2ind,
        "ind_to_class": ind2classes,
        "predicate_to_ind": predicates2ind,
        "ind_to_predicate": ind2predicates,
    }
    # with open(save_file, 'w') as f:
    #     json.dump(final_save, f)
    print("saved images = %d" % len(save_info))                                 # 46164     10000
    print("average boxes per image = %f" % ((box_count*1.0)/len(save_info)))    # 12.48     12.49
    print("saved relationships = %f" % ((pred_count*1.0)/len(save_info)))       # 9.12      9.17
    print("zero shot count = {}".format(zs_count))                              # 1523
    print("done.")
    print(len(classes2ind), len(predicates2ind))
    return exist_rels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="/hdd/sda/datasets/vrd/vg_msdn/", type=str)
    parser.add_argument('--save_path', default="./data/vg/", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    json_path = args.data_path
    img_path = "/hdd/sda/datasets/vrd/vg/images/"
    save_path = "./data/vg/vg_msdn/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #class_to_ind, ind_to_class = get_inds(os.path.join(data_path, 'objects.json'))
    #predicate_to_ind, ind_to_predicate = get_inds(os.path.join(data_path, 'predicates.json'))

    exist_rels = preprocess(os.path.join(json_path, 'train.json'), img_path,
               os.path.join(save_path, "train.json",),
               )
    preprocess(os.path.join(json_path, 'test.json'), img_path,
            os.path.join(save_path, "test.json"),
            exist_rels=exist_rels)

