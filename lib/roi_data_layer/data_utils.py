import numpy as np
from fast_rcnn.config import cfg

def cal_graph_matrix(num_roi, num_rel, relations):
    if num_roi <= 1: obj_d = np.eye(num_roi) * 1.0
    else: obj_d = np.eye(num_roi) * (1.0/(num_roi-1))
    obj_w = np.ones([num_roi, num_roi], dtype=float) - np.eye(num_roi)
    obj_m = np.matmul(np.matmul(np.sqrt(obj_d), obj_w), np.sqrt(obj_d))

    rel_mask_in = np.zeros((num_roi, num_rel)).astype(np.bool)
    rel_mask_in = np.ones((num_roi, num_rel)).astype(np.bool)
    rel_mask_out = np.zeros((num_roi, num_rel)).astype(np.bool)
    for i, rel in enumerate(relations):
        rel_mask_out[rel[0], i] = True
        rel_mask_in[rel[1], i] = True
    rel_d = np.zeros([num_rel, num_rel], dtype=np.float)
    rel_w = np.zeros([num_rel, num_rel], dtype=np.float)
    # rel_w = np.eye(num_rel, dtype=np.float)
    for i, rel in enumerate(relations):
        sub_rels = np.where(rel_mask_in[rel[0]])[0].tolist()
        for r in sub_rels:
            # if r != i:
            rel_d[i][i] += 1.0
            rel_w[i][r] = 1.0
        obj_rels = np.where(rel_mask_in[rel[1]])[0].tolist()
        for r in obj_rels:
            # if r != i:
            rel_d[i][i] += 1.0
            rel_w[i][r] = 1.0
    rel_d = 1.0/np.sqrt(rel_d)
    rel_d[np.isinf(rel_d)] = 0.0
    rel_m = np.matmul(np.matmul(rel_d, rel_w), rel_d)

    # print(num_roi, num_rel, relations)
    # print(obj_d)
    # print(obj_w)
    # print("rel_d", rel_d)
    # print("rel_w", rel_w)
    # print("rel_m", rel_m)
    # exit()

    return obj_m, rel_m

def cal_rel_weights(ims, rels):
    rel_weight_rois = np.array(
        [[im_i, 0.0, 0.0, ims.shape[2], ims.shape[1]] for im_i in range(len(ims))], dtype=np.float32)
    rel_weight = np.zeros([rels.shape[0]], dtype=np.int32)
    no_zeros_inds = np.where(rels[:, 2])
    rel_weight[no_zeros_inds] = 1.0
    return rel_weight, rel_weight_rois

def cal_rel_triples(rels, batch=32):
    rel_triple_inds = []
    rel_triple_labels = []
    pos_inds = np.where(rels[:, 2])[0]
    neg_inds = np.where(rels[:, 2]==0)[0]
    min_value=1
    if len(pos_inds)<=min_value or len(neg_inds)<=min_value:
        return np.zeros([0,2], np.int32), np.zeros([0,1], np.int32)

    rand_pos_inds = np.random.permutation(pos_inds)
    rand_neg_inds = np.random.permutation(neg_inds)

    size=0
    pos_i=0
    while(size<batch):
        # pp_i + a random pos-neg pair
        neg_i = np.random.choice(neg_inds)
        rel_triple_inds.append([rand_pos_inds[pos_i], neg_i])
        pos_i=(pos_i+1)%len(rand_pos_inds)
        size+=1

    rel_triple_inds = np.array(rel_triple_inds)
    rel_triple_inds[batch//2:, :] = rel_triple_inds[batch//2:, ::-1]
    rel_triple_labels = np.concatenate((np.tile([[1]],[batch//2,1]), np.tile([[0]],[batch//2,1])), axis=0)

    # print(rel_triple_inds)
    # print(rel_triple_labels)

    return rel_triple_inds, rel_triple_labels

def create_graph_data(num_roi, num_rel, relations):
    """
    compute graph structure from relations
    """

    # rel_fully_connect = True
    #
    # rel_mask = np.zeros((num_roi, num_rel)).astype(np.bool)
    # roi_rel_inds = np.ones((num_roi, num_roi)).astype(np.int32) * -1
    # for i, rel in enumerate(relations):
    #     rel_mask[rel[0], i] = True
    #     rel_mask[rel[1], i] = True
    #     roi_rel_inds[rel[0], rel[1]] = i
    #
    # # roi context
    # obj_context_o = []
    # obj_context_p = []
    # obj_context_inds = []
    # for i, mask in enumerate(rel_mask):
    #     rels = np.where(mask)[0].tolist()
    #     for reli in rels:
    #         obj_context_p.append(reli)
    #         o = relations[reli][0] if relations[reli][0]!=i else relations[reli][1]
    #         obj_context_o.append(o)
    #         obj_context_inds.append(i)
    #     obj_context_o.append(num_roi)
    #     obj_context_p.append(num_rel)
    #     obj_context_inds.append(i)
    #
    # # rel context
    # rel_context = []
    # rel_context_inds = []
    # for i, rel in enumerate(relations):
    #     if rel_fully_connect:
    #         for r in range(len(relations)):
    #             if r!=i:
    #                 rel_context.append(r)
    #                 rel_context_inds.append(i)
    #     else:
    #         sub_rels = np.where(rel_mask[rel[0]])[0].tolist()
    #         for r in sub_rels:
    #             if r!=i:
    #                 rel_context.append(r)
    #                 rel_context_inds.append(i)
    #         obj_rels = np.where(rel_mask[rel[1]])[0].tolist()
    #         for r in obj_rels:
    #             if r != i:
    #                 rel_context.append(r)
    #                 rel_context_inds.append(i)
    #     rel_context.append(num_rel)
    #     rel_context_inds.append(i)


    output_dict = {
        # 'obj_context_o': np.array(obj_context_o).astype(np.int32),
        # 'obj_context_p': np.array(obj_context_p).astype(np.int32),
        # 'obj_context_inds': np.array(obj_context_inds).astype(np.int32),
        # 'rel_context': np.array(rel_context).astype(np.int32),
        # 'rel_context_inds': np.array(rel_context_inds).astype(np.int32),
        'num_roi': num_roi,
        'num_rel': num_rel
    }

    return output_dict


def compute_rel_rois(num_rel, rois, relations):
    """
    union subject boxes and object boxes given a set of rois and relations
    """
    rel_rois = np.zeros([num_rel, 5])
    for i, rel in enumerate(relations):
        sub_im_i = rois[rel[0], 0]
        obj_im_i = rois[rel[1], 0]
        assert(sub_im_i == obj_im_i)
        rel_rois[i, 0] = sub_im_i

        sub_roi = rois[rel[0], 1:]
        obj_roi = rois[rel[1], 1:]
        union_roi = [np.minimum(sub_roi[0], obj_roi[0]),
                    np.minimum(sub_roi[1], obj_roi[1]),
                    np.maximum(sub_roi[2], obj_roi[2]),
                    np.maximum(sub_roi[3], obj_roi[3])]
        rel_rois[i, 1:] = union_roi

    return rel_rois
