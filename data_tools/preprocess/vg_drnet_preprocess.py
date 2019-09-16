"""
preprocessing vrd dataset into npy
"""
import os, json, argparse
import cv2
import pprint
import numpy as np

classes2ind = {}
predicates2ind = {}

def update_cls_predicate(phrase):
    global classes2ind, predicates2ind
    if phrase[0] in classes2ind:
        subid = classes2ind[phrase[0]]
    else:
        subid = len(classes2ind)
        classes2ind[phrase[0]] = subid
    if phrase[2] in classes2ind:
        objid = classes2ind[phrase[2]]
    else:
        objid = len(classes2ind)
        classes2ind[phrase[2]] = objid
    if phrase[1] in predicates2ind:
        predid = predicates2ind[phrase[1]]
    else:
        predid = len(predicates2ind)
        predicates2ind[phrase[1]] = predid
    return subid, predid, objid

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
        sub_ind, pred_ind, obj_ind = update_cls_predicate(rel['phrase'])
        sub = {'category': sub_ind, 'bbox': rel['subject']}
        obj = {'category': obj_ind, 'bbox': rel['object']}
        sub_id = update_objs(sub, objs)
        obj_id = update_objs(obj, objs)
        if [sub_id, obj_id, pred_ind] not in relations: relations.append([sub_id, obj_id, pred_ind])
    boxes, labels = [], []
    for obj in objs:
        box = obj['bbox']
        boxes.append(box)
        labels.append(obj['category'])
    return boxes, labels, relations

def preprocess(json_file, ims_path, save_file, exist_rels=None, num_classes=399, num_predicates=24):
    global predicates2ind, classes2ind
    is_train = 'train' in json_file
    info = json.load(open(json_file))
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
        filename = str(d['id']) + '.jpg'
        im_file = os.path.join(ims_path, filename)
        im = cv2.imread(im_file)
        if im is None:
            if os.path.exists(im_file):
                print("destroyed image: %s id: %d" % (im_file, i))
            else:
                print("missing image: %s id: %d. (you can try to change the img file from .gif to .jpg to fix this.)" % (im_file, i))
            continue
        height, width = im.shape[0], im.shape[1]
        boxes, labels, rels = get_objs_rels(d['relationships'])

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
    with open(save_file, 'w') as f:
        json.dump(final_save, f)
    print("saved images = %d" % len(save_info))                                 # 67086     954
    print("average boxes per image = %f" % ((box_count*1.0)/len(save_info)))    # 26430    6728      12.5
    print("saved relationships = %f" % ((pred_count*1.0)/len(save_info)))         # 30355    7632      9.2
    print("zero shot count = {}".format(zs_count))                              # 9
    print("done.")
    print(len(classes2ind), len(predicates2ind))                                  # 399  24
    return exist_rels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="/hdd/sda/datasets/vrd/vg_drnet/", type=str)
    parser.add_argument('--save_path', default="./data/vg/", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    json_path = args.data_path
    img_path = "/hdd/sda/datasets/vrd/vg/images/"
    save_path = "./data/vg/vg_drnet/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #class_to_ind, ind_to_class = get_inds(os.path.join(data_path, 'objects.json'))
    #predicate_to_ind, ind_to_predicate = get_inds(os.path.join(data_path, 'predicates.json'))

    exist_rels = preprocess(os.path.join(json_path, 'svg_train.json'), img_path,
               os.path.join(save_path, "train.json",),
               )
    preprocess(os.path.join(json_path, 'svg_test.json'), img_path,
            os.path.join(save_path, "test.json"),
            exist_rels=exist_rels)
