"""Functions for building Siamese Neural Networks [1,2,3]_.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: May 2018

Notes
-----

Siamese shared variable code based on:
- https://github.com/kamperh/tflego/blob/master/tflego/siamese.py.

References
----------
.. [1] F. Schroff, D. Kalenichenko, J. Philbin (2015):
        FaceNet: A Unified Embedding for Face Recognition and Clustering.
        https://arxiv.org/abs/1503.03832
.. [2] H. O. Song, Y. Xiang, S. Jegelka, S. Savarese (2015):
        Deep Metric Learning via Lifted Structured Feature Embedding.
        https://arxiv.org/abs/1511.06452
.. [3] A. Hermans, L. Beyer, B. Leibe (2017):
        In Defense of the Triplet Loss for Person Re-Identification.
        https://arxiv.org/abs/1703.07737
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


# ==============================================================================
#                                SIAMESE NETWORKS                               
# ------------------------------------------------------------------------------

def siamese_network(
        x_left_input,
        x_right_input,
        model_func,
        model_param_dict,
        input_param='x_input',
        var_scope='siamese'):
    """Build a Siamese tied network using `model_func` to define the left and right branches."""
    # Copy param dict for the duplicate left and right branches 
    model_l_param_dict = model_param_dict.copy()
    model_r_param_dict = model_param_dict.copy()
    model_l_param_dict[input_param] = x_left_input
    model_r_param_dict[input_param] = x_right_input

    # Build and return the Siamese model branches
    with tf.variable_scope(var_scope) as scope:
        x_out_left = model_func(**model_l_param_dict)
        scope.reuse_variables()
        x_out_right = model_func(**model_r_param_dict)
    return (x_out_left, x_out_right)


def siamese_triplet_network(
        x_anchor_input,
        x_same_input,
        x_different_input,
        model_func,
        model_param_dict,
        input_param='x_input',
        var_scope='siamese_triplet'):
    """Build a Siamese Triplet tied network using `model_func` to define the triplet branches."""
    # Copy param dict for the duplicate branches 
    model_a_param_dict = model_param_dict.copy()
    model_s_param_dict = model_param_dict.copy()
    model_d_param_dict = model_param_dict.copy()
    model_a_param_dict[input_param] = x_anchor_input
    model_s_param_dict[input_param] = x_same_input
    model_d_param_dict[input_param] = x_different_input

    # Build and return the Siamese Triplet model branches
    with tf.variable_scope(var_scope) as scope:
        x_out_anchor = model_func(**model_a_param_dict)
        scope.reuse_variables()
        x_out_same = model_func(**model_s_param_dict)
        x_out_different = model_func(**model_d_param_dict)
    return (x_out_anchor, x_out_same, x_out_different)


# ==============================================================================
#                                SIAMESE LOSS FUNCTIONS                               
# ------------------------------------------------------------------------------

def cos_similarity(x_1, x_2):
    l2_norm = lambda x: tf.norm(x, ord=2, axis=-1)
    return tf.reduce_sum(tf.multiply(x_1, x_2), -1) / (l2_norm(x_1) * l2_norm(x_2))


def cos_distance(x_1, x_2):
    return (1 - cos_similarity(x_1, x_2)) / 2.0


def loss_triplets_cos(
        x_anchor,
        x_same,
        x_diff,
        margin,
        average_non_zero=False,
        epsilon=1e-16):
    """Calculcate Triplet Loss [1]_ of the Siamese Triplet tied network.
    
    TODO(rpeloff) NOTE: 
    - Params `x_anchor` and `x_same` are same type, while `x_diff` is different
    - Enable "batch all" non-zero loss averaging with `average_non_zero=True`
    - Param `epsilon` determines which values are zero (i.e. losses < 1e-16 = 0)

    Notes
    -----
    The number of possible triplets grows cubically with the size of the 
    dataset. Most triplets are also learned by the network after a few epochs,
    making them uninformative. One solution is to mine "hard triplets", however
    a balance needs to be struck with "moderate negatives" since choosing only 
    the hardest triplets could select mostly outliers [1]_. 
    
    These negative exemplars are called semi-hard since they are further from 
    the anchor than the positive exemplar, but close to the anchor-positive 
    distance (see tensorflow for semi-hard online mining loss [1,4]_; also see
    the tensorflow lifted structure loss [3,5]_).

    A more recent solution is to follow a "Batch Hard" or "Batch All" strategy. 
    The "Batch All" non-zero strategy is the simplest modification to classic 
    triplet loss with competitive performance to other strategies. The idea is 
    to form batches of B=(P.K) concepts, where P is number of classes and K is 
    the number of examples per class (previous work suggests P=32 K=4), and 
    select all combinaton of triplets (which produces PK anchors, K-1 positives
    per anchor, PK-K negatives, and a total of PK(K-1)(PK-K) triplets). Since
    many of the triplets will be zero, "washing out" the few useful triplets, we
    average only the non-zero loss terms (L_BA/=0 in [3]_). 

    The non-zero batch-all strategy can be enabled with `average_non_zero=False`
    AND by sampling triplets from batch B=P.K as mentioned above (Should be 
    PK(K-1)(PK-K) anchors with corresponding same and diff pairs).
    NOTE: This is inefficient and pairwise online mining should be used!

    References
    ----------
    .. [1] F. Schroff, D. Kalenichenko, J. Philbin (2015):
            FaceNet: A Unified Embedding for Face Recognition and Clustering.
            https://arxiv.org/abs/1503.03832
    .. [2] H. O. Song, Y. Xiang, S. Jegelka, S. Savarese (2015):
            Deep Metric Learning via Lifted Structured Feature Embedding.
            https://arxiv.org/abs/1511.06452
    .. [3] A. Hermans, L. Beyer, B. Leibe (2017):
            In Defense of the Triplet Loss for Person Re-Identification.
            https://arxiv.org/abs/1703.07737
    .. [4] TensorFlow contrib metric-learning losses (Accessed: August 2018):
            Computes the triplet loss with semi-hard negative mining.
            https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss
    .. [5] TensorFlow contrib metric-learning losses (Accessed: August 2018): 
            Computes the lifted structured loss.
            https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/lifted_struct_loss
    .. [6] O. Moindrot (2018):
            Triplet Loss and Online Triplet Mining in TensorFlow.
            https://omoindrot.github.io/triplet-loss

    Examples
    --------
    >>> import mltoolset as ml
    >>> import tensorflow as tf
    >>> a = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> s = tf.constant([[4., 5., 6.], [7., 8., 9.], [1., 2., 3.]])
    >>> d = tf.constant([[-1., -2., -3.], [-4., -5., -6.], [4., 5., 6.]])
    >>> l = ml.siamese.loss_triplets_cos(a, s, d, 0.2)
    >>> l_non_zero, frac = ml.siamese.loss_triplets_cos(a, s, d, 0.2, average_non_zero=True)
    >>> sess = tf.Session()
    >>> sess.run(l)
    0.07312984

    First two triplets are easy with 0. loss, and third is more difficult with 0.219 loss. 
    However, triplet loss is low since we average over all three loss terms.

    >>> sess.run([l_non_zero, frac])
    [0.21938951, 0.33333334]

    Loss has now increased since we average over only the one non-zero term (i.e. 33% of terms).

    >>> sess.close()
    """
    # Compute triplet losses
    pos_dists = cos_distance(x_anchor, x_same)
    neg_dists = cos_distance(x_anchor, x_diff)
    loss_dists = margin + pos_dists - neg_dists
    losses = tf.maximum(0., loss_dists)
    # Add some tensorboard summaries
    tf.summary.histogram('siamese_pos_dists', pos_dists)
    tf.summary.histogram('siamese_neg_dists', neg_dists)
    tf.summary.histogram('siamese_loss_dists', loss_dists)
    if not average_non_zero:
        # Return classical triplet loss
        return tf.reduce_mean(losses)
    else:       
        # Count the number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.to_float(tf.greater(losses, epsilon))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.to_float(tf.shape(losses)[0])       
        # Get average of non-zero triplet loss terms, and fraction positive
        triplet_loss = tf.reduce_sum(losses) / (num_positive_triplets + epsilon)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + epsilon)
        tf.summary.scalar('frac_positive_triplets', fraction_positive_triplets)
        # Return triplet loss with only non-zero loss terms (Sample batches B=P.K !)
        return triplet_loss
