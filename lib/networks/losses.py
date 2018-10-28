import tensorflow as tf

def sparse_softmax(logits, labels, name, loss_weight=1, ignore_bg=False):
    labels = tf.cast(labels, dtype=tf.int32)
    batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    if ignore_bg: # do not penalize background class
        loss_mask = tf.cast(tf.greater(labels, 0), tf.float32)
        batch_loss = tf.multiply(batch_loss, loss_mask)
    loss = tf.reduce_mean(batch_loss)
    loss = tf.multiply(loss, loss_weight, name=name)
    return loss

def l1_loss(preds, targets, name, target_weights=None, loss_weight=1):
    l1 = tf.abs(tf.subtract(preds, targets))
    if target_weights is not None:
        l1 = tf.multiply(target_weights, l1)
    batch_loss = tf.reduce_sum(l1, axis=1)
    loss = tf.reduce_mean(batch_loss)
    loss = tf.multiply(loss, loss_weight, name=name)
    return loss

def exp_average_summary(ops, dep_ops, decay=0.9, name='avg', scope_pfix='',
                        raw_pfix=' (raw)', avg_pfix=' (avg)'):
    averages = tf.train.ExponentialMovingAverage(decay, name=name)
    averages_op = averages.apply(ops)

    for op in ops:
        tf.summary.scalar(scope_pfix + op.name + raw_pfix, op)
        tf.summary.scalar(scope_pfix + op.name + avg_pfix,
                          averages.average(op))

    with tf.control_dependencies([averages_op]):
        for i, dep_op in enumerate(dep_ops):
            dep_ops[i] = tf.identity(dep_op, name=dep_op.name.split(':')[0])

    return dep_ops

def total_loss_and_summaries(losses, name):
    total_loss = tf.add_n(losses, name=name)
    losses.append(total_loss)
    total_loss = exp_average_summary(losses, [total_loss],
                                     decay=0.9, name='losses_avg',
                                     scope_pfix='losses/')[0]
    return total_loss


def accuracy(pred, labels, name, ignore_bg=False):
    correct_pred = tf.cast(tf.equal(labels, tf.cast(pred, tf.int32)), tf.float32)
    if ignore_bg:  # ignore background
        mask = tf.cast(tf.greater(labels, 0), tf.float32)
        one = tf.constant([1], tf.float32)
        # in case zero foreground preds
        num_preds = tf.maximum(tf.reduce_sum(mask), one)
        acc_op = tf.squeeze(tf.div(tf.reduce_sum(tf.multiply(correct_pred, mask)), num_preds))
    else:
        acc_op = tf.reduce_mean(correct_pred, tf.float32)

    # override the name
    return tf.identity(acc_op, name=name)
