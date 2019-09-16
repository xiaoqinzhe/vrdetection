"""
preprocessing vrd dataset into npy
"""
import os, json, argparse
import cv2, h5py
import pprint
import numpy as np

classes2ind = {}
predicates2ind = {}

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

def get_objs_rels(d):
    objs = []
    relations = []
    tri_labels = d['rlp_labels'][:]
    sub_boxes = d['sub_boxes'][:]
    obj_boxes = d['obj_boxes'][:]
    for i in range(len(tri_labels)):
        sub_ind, pred_ind, obj_ind = int(tri_labels[i][0])-1, int(tri_labels[i][1]), int(tri_labels[i][2]-1)
        assert sub_ind>=0 and obj_ind >= 0
        sub = {'category': sub_ind, 'bbox': sub_boxes[i].tolist()}
        obj = {'category': obj_ind, 'bbox': obj_boxes[i].tolist()}
        sub_id = update_objs(sub, objs)
        obj_id = update_objs(obj, objs)
        if [sub_id, obj_id, pred_ind] not in relations: relations.append([sub_id, obj_id, pred_ind])
    boxes, labels = [], []
    for obj in objs:
        box = obj['bbox']
        boxes.append(box)
        labels.append(obj['category'])
    return boxes, labels, relations

def preprocess(json_file, ims_path, save_file, exist_rels=None, num_classes=200, num_predicates=100):
    global predicates2ind, classes2ind
    is_train = exist_rels is None
    info = h5py.File(json_file, 'r')
    # 'gt' -> 'train', 'test' -> 'id' -> 'sub_boxes', 'obj_boxes', 'rlp_labels'
    # 'meta' -> 'cls', 'imid2path', 'pre' -> ('idx2name', 'name2idx'), ('id')

    import pprint
    #a=info['gt']['train']['999']
    print(info['meta'].keys())
    if len(predicates2ind) == 0:
        for name in info['meta']['pre']['name2idx'].keys():
            predicates2ind[name] = int(info['meta']['pre']['name2idx'][name][...])
        for name in info['meta']['cls']['name2idx'].keys():
            if name == "__background__": continue
            classes2ind[name] = int(info['meta']['cls']['name2idx'][name][...])-1
        print(predicates2ind)
        print(classes2ind)

    save_info = []
    if is_train:
        info = info['gt']['train']
    else: info = info['gt']['test']
    print("images amount = %d"%len(info))
    box_count, pred_count = 0, 0
    zs_count=0
    # recording rels that exists in training data
    if is_train:
        exist_rels = np.zeros([num_classes, num_classes, num_predicates], dtype=np.bool)
    j = 0
    for i in info.keys():
        # if j>10: break;
        d = info[i]
        j += 1
        if j % 1000 == 0:
            print("processing %s"%j)
        filename = i + '.jpg'
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

        if(len(boxes)>50): print(i)

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
    #with open(save_file, 'w') as f:
    #    json.dump(final_save, f)
    print("saved images = %d" % len(save_info))                                 # 73794     25858
    print("average boxes per image = %f" % ((box_count*1.0)/len(save_info)))    # 12.97    13.23
    print("saved relationships = %f" % ((pred_count*1.0)/len(save_info)))       # 9.23    9.43
    print("zero shot count = {}".format(zs_count))                              # 5632
    print("done.")
    print(len(classes2ind), len(predicates2ind))                                # 200 100
    return exist_rels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="/hdd/sda/datasets/vrd/vg_vtranse/", type=str)
    parser.add_argument('--save_path', default="./data/vg/", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    json_path = args.data_path
    img_path = "/hdd/sda/datasets/vrd/vg/images/"
    save_path = "./data/vg/vg_vtranse/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #class_to_ind, ind_to_class = get_inds(os.path.join(data_path, 'objects.json'))
    #predicate_to_ind, ind_to_predicate = get_inds(os.path.join(data_path, 'predicates.json'))

    exist_rels = preprocess(os.path.join(json_path, 'vg1_2_meta.h5'), img_path,
               os.path.join(save_path, "train.json",),
               )
    preprocess(os.path.join(json_path, 'vg1_2_meta.h5'), img_path,
            os.path.join(save_path, "test.json"),
            exist_rels=exist_rels)

