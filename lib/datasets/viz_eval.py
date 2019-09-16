import numpy as np
from fast_rcnn.config import cfg
from utils.visualization import draw_image
import os
total = 0

def get_oo_id(i, j, num_class):
    return i*num_class + j

def viz_relation(sg_entry, imdb,
                         im_i,
                         result_dict, prior=None
                         ):

    roidb = imdb.roidb[im_i]
    predicate_preds = sg_entry['relations']
    rel_weights = sg_entry["rel_weights"]
    # no bg
    if cfg.TRAIN.USE_SAMPLE_GRAPH:
        predicate_preds = predicate_preds[:, :, 1:]

    # use prediction, prior, weight
    ind2class = imdb.ind_to_classes
    ind2predicate = imdb.ind_to_predicates

    filename = os.path.basename(imdb.info[im_i]['image_filename'])
    print("------------------\nimage_name: {}".format(filename))

    # draw image
    im_path = imdb.get_image_path(im_i)
    save_path = "./data/"+cfg.DATASET+"/images_det/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path+filename):
        draw_image(im_path, roidb["boxes"], roidb["gt_classes"], ind2class, save_path+filename)

    print("boxes:", roidb['boxes'])
    print("labels", imdb.ind_to_classes[roidb["gt_classes"]])
    print("relationship:")
    num_boxes = len(roidb["boxes"])
    gt_i = [[[] for i in range(num_boxes)] for j in range(num_boxes)]
    gt_op = [[[] for i in range(num_boxes)] for j in range(num_boxes)]
    gt_relations = roidb["gt_relations"]
    gt_classes = roidb['gt_classes']
    zs_tags = roidb['zero_shot_tags']
    num_classes = len(ind2class)
    for i, rel in enumerate(roidb["gt_relations"]): gt_op[rel[0]][rel[1]].append(rel[2])
    for i, rel in enumerate(roidb["gt_relations"]): gt_i[rel[0]][rel[1]].append(i)
    # print(num_boxes, len(gt_op), len(gt_op[0]))
    k=0
    for i in range(num_boxes):
        for j in range(num_boxes):
            if i==j: continue
            # print(i,j)
            if len(gt_op[i][j]) > 0:
                # gt
                print("gt_rel: <{}:{}, {}:{}, {}:{}>".format(
                    i, ind2class[roidb["gt_classes"][i]],
                    gt_op[i][j], [ind2predicate[pi] for pi in gt_op[i][j]],
                    j, ind2class[roidb["gt_classes"][j]])
                )
                # prediction
                gt_p = gt_op[i][j]
                p = np.argmax(predicate_preds[i][j])
                p_v = np.max(predicate_preds[i][j])
                p_pri = np.argmax(predicate_preds[i][j]*prior[get_oo_id(gt_classes[i], gt_classes[j],num_classes)])
                p_pri_v = np.max(predicate_preds[i][j]*prior[get_oo_id(gt_classes[i], gt_classes[j],num_classes)])
                p_att_v = p_pri_v * rel_weights[k]
                zs = False
                for z in gt_i[i][j]: zs = (zs or zs_tags[z])
                if cfg.TRAIN.USE_SAMPLE_GRAPH:
                    p+=1
                    p_pri+=1
                stat = ""
                if zs: stat+="zs: "
                p_b = False
                for u in gt_p: p_b = p_b or (u==p)
                pp_b = False
                for u in gt_p: pp_b = pp_b or (u==p_pri)
                if not p_b and pp_b: stat += "improved: "
                stat += "{} ({}), ".format(ind2predicate[p], p_b)
                stat += "{} ({}), ".format(ind2predicate[p_pri], pp_b)
                print(stat)
                # salient weight
                print("weight:", rel_weights[k], "3: ", p_v, p_pri_v, p_att_v)
            else:
                p = np.argmax(predicate_preds[i][j])
                p_v = np.max(predicate_preds[i][j])
                p_pri = np.argmax(predicate_preds[i][j] * prior[get_oo_id(gt_classes[i], gt_classes[j], num_classes)])
                p_pri_v = np.max(predicate_preds[i][j] * prior[get_oo_id(gt_classes[i], gt_classes[j], num_classes)])
                p_att_v = p_pri_v * rel_weights[k]
                if cfg.TRAIN.USE_SAMPLE_GRAPH:
                    p+=1
                    p_pri+=1
                print("bg_rel: <{}:{}, {}:{}({}), {}:{}>".format(
                    i, ind2class[roidb["gt_classes"][i]],
                    p, ind2predicate[p], ind2predicate[p_pri],
                    j, ind2class[roidb["gt_classes"][j]]))
                print("weight:", rel_weights[k], "3: ", p_v, p_pri_v, p_att_v)
            k+=1
