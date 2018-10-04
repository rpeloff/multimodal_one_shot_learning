"""Functions for performing efficient k-Nearest Neighbours search on a GPU.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: August 2018

Notes
-----

Based on (exact mode) efficient nearest neighbours computation proposed by [1]_. 

References
----------
.. [1] L. Kaiser, O. Nachum, A. Roy, S. Bengio (2017):
        Learning to Remember Rare Events.
        https://arxiv.org/abs/1703.03129
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf



def fast_knn_cos(q_batch, m_keys, k_nn=1, normalize=True):
    """Fast kNN implementation based on cosine similarity.

    NOTE:
    - Set `normalize=False` only if `q_batch` and `m_keys` are already normalized.

    Notes
    -----

    Implementation of the exact mode efficient nearest neighbor computation
    described by [1]_.

    For each query q (in `q_batch`), the nearest neighbour in memory M is
    computed as (where K[i] are the keys in memory `m_keys`):

        NN(q,M) = argmax_i q·K[i]

    Since the query and memory keys are *normalized (|q|=1; |K[i]|=1), this
    corresponds to nearest neighbour with cosine similarity.

    The nearest neighbours for the batch of queries Q (`q_batch`) is computed
    with a single matrix multiplication: Q x M^T

    (*normalized: beforehand or by this function if `normalize=True`)

    References
    ----------
    .. [1] L. Kaiser, O. Nachum, A. Roy, S. Bengio (2017):
            Learning to Remember Rare Events.
            https://arxiv.org/abs/1703.03129
    """
    if normalize:
        q_batch = tf.nn.l2_normalize(q_batch, axis=1)
        m_keys = tf.nn.l2_normalize(m_keys, axis=1)
    
    cos_similarities = tf.matmul(q_batch, tf.transpose(m_keys))

    nearest_ind = lambda: tf.argmax(cos_similarities, axis=1, output_type=tf.int32)
    k_nearest_ind = lambda: tf.nn.top_k(cos_similarities, k=k_nn)[1]

    return tf.cond(tf.equal(k_nn, 1),
                   true_fn=nearest_ind,
                   false_fn=k_nearest_ind)
