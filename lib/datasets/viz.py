# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

from fast_rcnn.config import cfg
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph

plt.switch_backend("agg")

"""
Utility for visualizing a scene graph
"""

def draw_scene_graph(labels, inds, rels, filename):
    """
    draw a graphviz graph of the scene graph topology
    """
    viz_labels = labels[inds]
    viz_rels = None
    if rels is not None:
        viz_rels = []
        for rel in rels:
            if rel[0] in inds and rel[1] in inds :
                sub_idx = np.where(inds == rel[0])[0][0]
                obj_idx = np.where(inds == rel[1])[0][0]
                viz_rels.append([sub_idx, obj_idx, rel[2]])
    return draw_graph(viz_labels, viz_rels, cfg, filename)


def draw_graph(labels, rels, cfg, filename):
    u = Digraph('sg', filename='sg.gv')
    u.body.append('size="6,6"')
    u.body.append('rankdir="LR"')
    u.node_attr.update(style='filled')

    out_dict = {'ind_to_class': cfg.ind_to_class, 'ind_to_predicate': cfg.ind_to_predicate}
    out_dict['labels'] = labels.tolist()
    out_dict['relations'] = rels

    rels = np.array(rels)
    rel_inds = rels[:,:2].ravel().tolist()
    name_list = []
    for i, l in enumerate(labels):
        if i in rel_inds:
            name = cfg.ind_to_class[l]
            name_suffix = 1
            obj_name = name
            while obj_name in name_list:
                obj_name = name + '_' + str(name_suffix)
                name_suffix += 1
            name_list.append(obj_name)
            u.node(str(i), label=obj_name, color='lightblue2')

    for rel in rels:
        edge_key = '%s_%s' % (rel[0], rel[1])
        u.node(edge_key, label=cfg.ind_to_predicate[rel[2]], color='red')

        u.edge(str(rel[0]), edge_key)
        u.edge(edge_key, str(rel[1]))

    # u.view()
    # u.save(filename+"_graph.jpg")

    return out_dict


def viz_scene_graph(im, rois, labels, filename, inds=None, rels=None, gt_rels=None, preprocess=True, imagename="image"):
    """
    visualize a scene graph on an image
    """
    if inds is None:
        inds = np.arange(rois.shape[0])
    viz_rois = rois[inds]
    viz_labels = labels[inds]
    viz_rels = None
    if rels is not None:
        viz_rels = []
        for rel in rels:
            if rel[0] in inds and rel[1] in inds :
                sub_idx = np.where(inds == rel[0])[0][0]
                obj_idx = np.where(inds == rel[1])[0][0]
                viz_rels.append([sub_idx, obj_idx, rel[2]])
        viz_rels = np.array(rels)
    return _viz_scene_graph(im, viz_rois, viz_labels, filename, viz_rels, gt_rels, preprocess, imagename=imagename)


def _viz_scene_graph(im, rois, labels, filename, rels=None, gt_rels=None, preprocess=True, imagename="image"):
    # print(rels)
    # print(gt_rels)
    if preprocess:
        # transpose dimensions, add back channel means
        im = (im.copy() + cfg.PIXEL_MEANS)[:, :, (2, 1, 0)].astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    if rels.size > 0:
        rel_inds = rels[:,:2].ravel().tolist()
    else:
        rel_inds = []
    # draw bounding boxes
    for i, bbox in enumerate(rois):
        if int(labels[i]) == 0 and i not in rel_inds:
            continue
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        label_str = cfg.ind_to_class[int(labels[i])]+"_"+str(i)
        ax.text(bbox[0], bbox[1] - 2,
                label_str,
                bbox=dict(facecolor='blue', alpha=0.3),
                fontsize=8, color='white')

    # draw relations
    old_rels = []
    gt_rel_objs = np.array([[rel[0], rel[1]] for rel in gt_rels])
    for i, rel in enumerate(rels):
        # if rel[2] == 0: # ignore bachground
        #    continue
        sub_box = rois[rel[0], :]
        obj_box = rois[rel[1], :]
        obj_ctr = [obj_box[0], obj_box[1] - 2]
        sub_ctr = [sub_box[0], sub_box[1] - 2]
        line_ctr = [(sub_ctr[0] + obj_ctr[0]) / 2, (sub_ctr[1] + obj_ctr[1]) / 2]

        is_correct = False
        gt_rel_i = -1
        for gt_rel in gt_rels:
            if rel[0]==gt_rel[0] and rel[1]==gt_rel[1]:
                if rel[2] == gt_rel[2]:
                    is_correct = True
                gt_rel_i = gt_rel[2]
        if is_correct:
            predicate = cfg.ind_to_predicate[int(rel[2])]
            p_color = 'green'
        else:
            predicate = cfg.ind_to_predicate[int(rel[2])] + " {}->{} ".format(rel[0], rel[1]) + cfg.ind_to_predicate[gt_rel_i]
            p_color = 'red'
        ax.arrow(sub_ctr[0], sub_ctr[1], obj_ctr[0]-sub_ctr[0], obj_ctr[1]-sub_ctr[1], color='white', head_width=5)

        if [rel[1], rel[0]] in old_rels:
            line_ctr[1] += 10
        ax.text(line_ctr[0], line_ctr[1], predicate,
                bbox=dict(facecolor=p_color, alpha=0.3),
                fontsize=8, color='white')
        old_rels.append([rel[0], rel[1]])

    ax.set_title(imagename, fontsize=10)
    ax.axis('off')
    fig.tight_layout()
    # plt.show()
    plt.savefig(filename+"_fig.jpg")
