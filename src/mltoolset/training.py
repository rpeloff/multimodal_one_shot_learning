"""Functions for training models.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: August 2018
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def get_training_op(
        optimizer_class, 
        loss_func, 
        learn_rate, 
        decay_rate, 
        n_epoch_batches):
    """Return a training operation for the specified loss and optimizer."""
    # Create variable that will be incremented by each minimize() step
    global_step = tf.train.get_or_create_global_step()

    # Decay learning rate by decay rate at the end of each epoch
    decay_learn_rate = tf.train.exponential_decay(
        learning_rate=learn_rate,
        global_step=global_step,
        decay_steps=n_epoch_batches,
        decay_rate=decay_rate,
        staircase=True)
    # Create tf.summary containing the (decayed) learning rate
    tf.summary.scalar('learning_rate', decay_learn_rate)

    # Initialize specified optimizer with the (decayed) learning rate
    optimizer = optimizer_class(learning_rate=decay_learn_rate)

    # Get update op dependencies and add them to the train op 
    # (e.g. `tf.layers.batch_normalization` adds mean and variance update ops)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        training_op = optimizer.minimize(loss_func,
                                            global_step=global_step)
    # Return training operation and global step counter
    return training_op


# TODO(rpeloff) move general functions from train scripts here ...
