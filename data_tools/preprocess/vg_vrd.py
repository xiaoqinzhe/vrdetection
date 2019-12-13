import json
from collections import Counter


def load_datasets(paths):
    trains, tests = [], []
    ind2preds, ind2class = [], []
    for path in paths:
        trains.append(json.load(open(path+'train.json')))
        tests.append(json.load(open(path + 'test.json')))
        ind2class.append(trains[-1]['ind_to_class'])
        ind2preds.append(trains[-1]['ind_to_predicate'])
    print(ind2class[0], '\n', ind2preds[0])
    common_classes, common_preds = [], []
    for i in range(len(ind2class[0])):
        cls = ind2class[0][i]
        flag = True
        for j in range(1, len(ind2class)):
            if cls not in ind2class[j]:
                flag = False
                break
        if flag: common_classes.append(cls)
    for i in range(len(ind2preds[0])):
        cls = ind2preds[0][i]
        flag = True
        for j in range(1, len(ind2preds)):
            if cls not in ind2preds[j]:
                flag = False
                break
        if flag: common_preds.append(cls)
    print(len(common_classes), common_classes, '\n', len(common_preds), sorted(common_preds))

def show():
    vrd = json.load(open("./data/vrd/train.json"))
    vrd_id2class = vrd['ind_to_class']
    vrd_id2pred = vrd['ind_to_predicate']

    # alias check...
    def make_alias_dict(dict_file):
        """create an alias dictionary from a file"""
        out_dict = {}
        vocab = []
        for line in open(dict_file, 'r'):
            alias = line.strip('\n').strip('\r').split(',')
            alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
            for a in alias:
                out_dict[a] = alias_target  # use the first term as the aliasing target
            vocab.append(alias_target)
        return out_dict, vocab
    class_alias_dict, _ = make_alias_dict("./data_tools/VG/object_alias.txt")
    predicate_alias_dict, _ = make_alias_dict("./data_tools/VG/predicate_alias.txt")

    for cls in vrd_id2class:
        if cls in class_alias_dict:
            print(cls, class_alias_dict[cls])
    for pred in vrd_id2pred:
        if pred in predicate_alias_dict:
            print(pred, predicate_alias_dict[pred])

    vg_obj_counter = {ele[0]: ele[1] for ele in json.load(open("./data/vg/obj_counter.json"))}
    vg_pred_counter = {ele[0]: ele[1] for ele in json.load(open("./data/vg/pred_counter.json"))}
    print(len(vg_obj_counter), len(vg_pred_counter))

    if "by" in vg_pred_counter: print(vg_pred_counter['by'])

    common_classes, common_preds = [], []
    for i in range(len(vrd_id2class)):
        cls = vrd_id2class[i]
        if cls in vg_obj_counter:
            common_classes.append((cls, vg_obj_counter[cls]))
    for i in range(len(vrd_id2pred)):
        pred = vrd_id2pred[i]
        if pred in vg_pred_counter and vg_pred_counter[pred] > 0:
            common_preds.append((pred, vg_pred_counter[pred]))
    print(len(common_classes), sorted(common_classes, key=lambda x: x[1]), '\n', len(common_preds), sorted(common_preds, key=lambda x: x[0]))

if __name__ == '__main__':
    vrd_path = './data/vrd/'
    vg_path = './data/vg/' + 'vg_vtranse/'
    load_datasets([vrd_path, vg_path])
    show()