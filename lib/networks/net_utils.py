import tensorflow as tf



def exp_average(vec, curr_avg, decay=0.9):
    vec_avg = tf.reduce_mean(vec, 0)
    avg = tf.assign(curr_avg, curr_avg * decay + vec_avg * (1-decay))
    return avg

def gather_vec_pairs(vecs, gather_inds):
    """
    gather obj-subj feature pairs
    """
    vec_pairs = tf.gather(vecs, gather_inds)
    vec_len = int(vec_pairs.get_shape()[2]) * 2
    vec_pairs = tf.reshape(vec_pairs, [-1, vec_len])
    return vec_pairs

def pad_and_gather(vecs, mask_inds, pad=None):
    """
    pad a vector with a zero row and gather with input inds
    """
    if pad is None:
        pad = tf.expand_dims(tf.zeros_like(vecs[0]), 0)
    else:
        pad = tf.expand_dims(pad, 0)
    vecs_padded = tf.concat([vecs, pad], 0)
    # flatten mask and edges
    vecs_gathered = tf.gather(vecs_padded, mask_inds)
    return vecs_gathered

def padded_segment_reduce(vecs, segment_inds, num_segments, reduction_mode):
    """
    Reduce the vecs with segment_inds and reduction_mode
    Input:
        vecs: A Tensor of shape (batch_size, vec_dim)
        segment_inds: A Tensor containing the segment index of each
        vec row, should agree with vecs in shape[0]
    Output:
        A tensor of shape (vec_dim)
    """
    if reduction_mode == 'max':
        print('USING MAX POOLING FOR REDUCTION!')
        vecs_reduced = tf.segment_max(vecs, segment_inds)
    elif reduction_mode == 'mean':
        print('USING AVG POOLING FOR REDUCTION!')
        vecs_reduced = tf.segment_mean(vecs, segment_inds)
    vecs_reduced.set_shape([num_segments, vecs.get_shape()[1]])
    return vecs_reduced
