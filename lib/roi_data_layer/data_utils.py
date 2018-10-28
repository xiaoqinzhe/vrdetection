import numpy as np
import gensim
from fast_rcnn.config import cfg


def load_word_vec(words):
    if cfg.WORD2VEC_FILE.endswith('.txt'):
        wordvec = gensim.models.KeyedVectors.load_word2vec_format(cfg.WORD2VEC_FILE, binary=False)
    else:
        wordvec = gensim.models.KeyedVectors.load_word2vec_format(cfg.WORD2VEC_FILE, binary=True)
    return [wordvec[w] for w in words]

def create_graph_data(num_roi, num_rel, relations):
    """
    compute graph structure from relations
    """

    rel_mask = np.zeros((num_roi, num_rel)).astype(np.bool)
    roi_rel_inds = np.ones((num_roi, num_roi)).astype(np.int32) * -1
    for i, rel in enumerate(relations):
        rel_mask[rel[0], i] = True
        rel_mask[rel[1], i] = True
        roi_rel_inds[rel[0], rel[1]] = i

    # roi context
    obj_context_o = []
    obj_context_p = []
    obj_context_inds = []
    for i, mask in enumerate(rel_mask):
        rels = np.where(mask)[0].tolist()
        for reli in rels:
            obj_context_p.append(reli)
            o = relations[reli][0] if relations[reli][0]!=i else relations[reli][1]
            obj_context_o.append(o)
            obj_context_inds.append(i)
        obj_context_o.append(num_roi)
        obj_context_p.append(num_rel)
        obj_context_inds.append(i)

    # rel context
    rel_context = []
    rel_context_inds = []
    for i, rel in enumerate(relations):
        sub_rels = np.where(rel_mask[rel[0]])[0].tolist()
        for r in sub_rels:
            if r!=i:
                rel_context.append(r)
                rel_context_inds.append(i)
        obj_rels = np.where(rel_mask[rel[1]])[0].tolist()
        for r in obj_rels:
            if r != i:
                rel_context.append(r)
                rel_context_inds.append(i)
        rel_context.append(num_rel)
        rel_context_inds.append(i)


    output_dict = {
        'obj_context_o': np.array(obj_context_o).astype(np.int32),
        'obj_context_p': np.array(obj_context_p).astype(np.int32),
        'obj_context_inds': np.array(obj_context_inds).astype(np.int32),
        'rel_context': np.array(rel_context).astype(np.int32),
        'rel_context_inds': np.array(rel_context_inds).astype(np.int32),
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
