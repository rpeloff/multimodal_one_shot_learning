"""
Tools for creating and batching datasets with TensorFlow.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2018

Notes
-----
When using functions to create tf.data.Dataset objects that will consume large
NumPy arrays, it is best practice to feed the NumPy data with placeholders when 
initializing an iterator. This avoids hitting memory limits of tf.GraphDef. 
For example:

>>> large_features = ...  # load large NumPy data features
>>> large_labels = ...  # load large NumPy data labels
>>> x_input = tf.placeholder(large_features.dtype, large_features.shape)
>>> y_input = tf.placeholder(large_labels.dtype, large_labels.shape)
>>> dataset = ml.data.batch_k_examples_for_p_concepts(x_input, y_labels, p, k)
>>> ...  # other dataset transformations
>>> ...  # build model on dataset
>>> iterator = dataset.make_initializable_iterator()
>>> ...  # setup tf.Session
>>> sess.run(iterator.initializer, feed_dict={x_input: large_features,
...                                           y_input: large_labels})
>>> while True:
...     try:
...         sess.run(iterator.get_next())
...     except tf.errors.OutOfRangeError:
...         break
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import imageio


from .. import _globals


def make_train_test_split(*data, test_ratio=0.2, shuffle=True, seed=42):
    """Split data arrays into train and test splits.
    
    Arrays are split together such that elements across the split data arrays
    correspond, hence they must all have the same length (i.e. shape[0]).
    
    Parameters
    ----------
    *data : seqeunce of array-like (with same shape[0])
        Data arrays (e.g. features, labels) to split (must have same length).
    test_ratio : float, optional
        Ratio (between 0.0 and 1.0) of the dataset to include in the test set (default is 0.2).
    shuffle : bool, optional
        Flag to shuffle the data before splitting (default is True).
    seed : int, optional
        Op-level seed for repeatable distributions across sessions (default is 42).
    
    Returns
    -------
    data_split : tuple, length=2*len(data)
        List of train and test splits for each data input.

    Notes
    -----
    If a tf.data.Dataset is created from data returned by make_train_test_split,
    then it cannot be iterated by a one-shot iterator since tf.random_shuffle
    creates a stateful node that needs to be initialized.
    """
    # TODO(rpeloff) add control dependency to check data has same shape[0]
    if test_ratio < 0.0 or test_ratio > 1.0:
        raise ValueError("Test size must be between 0.0 and 1.0. Got: {}".format(test_ratio))
    if seed is None:  # make sure values in data shuffled by same op-level seed
        seed = 42
    data_split = ()
    for d in data:
        # Get train/test size and split data into train/test tensors
        d_value = tf.convert_to_tensor(d)
        d_shape = tf.cast(tf.shape(d_value)[0], _globals.TF_FLOAT)
        train_size, test_size = (
            tf.cast(tf.ceil(tf.multiply(d_shape, 1.-test_ratio)), _globals.TF_INT),
            tf.cast(tf.floor(tf.multiply(d_shape, test_ratio)), _globals.TF_INT))
        if shuffle:  # shuffle data tensor if specified
            d_value = tf.random_shuffle(d_value, seed=seed)
        d_train, d_test = tf.split(d_value,
                                   [train_size, test_size])
        data_split += ((d_train, d_test),)
    return data_split


def batch_dataset(
        x_data,
        y_labels,
        batch_size,
        shuffle=True,
        drop_remainder=True):
    """Create mini-batch data pipeline that iterates over the full dataset."""
    n_data = tf.shape(x_data, out_type=tf.int64)[0]
    batched_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(x_data),
        tf.data.Dataset.from_tensor_slices(y_labels)))
    if shuffle:
        batched_dataset = batched_dataset.shuffle(n_data)
    if drop_remainder:  # TODO(rpeloff) TF 1.10 this is part of dataset.batch()
        batched_dataset = batched_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
        batched_dataset = batched_dataset.batch(batch_size)
    return batched_dataset


def batch_k_examples_for_p_concepts(
        x_data,
        y_labels,
        p_batch,
        k_batch #, unique_concepts=None
        ):
    """Create dataset with batches of P concept classes and K examples per class.   
    
    Used to produce balanced mini-batches of PK examples, by randomly sampling 
    P concept labels, and then K examples per concept [1]_. 

    Parameters
    ----------
    x_data : array-like or tf.Tensor
        Dataset of concept features.
    y_labels : array-like or tf.Tensor
        Dataset of concept labels associated with features. 
    p_batch : int
        Number of P unique concept labels to sample per batch.
    k_batch : int
        Number of K examples to sample per unique concept in a batch.
    unique_concepts : array-like, optional
        List of unique concept classes from which P concepts are sampled.

    Returns
    -------
    balanced_dataset : tf.data.Dataset
        Balanced dataset containing batches (x_batch, y_batch) of PK examples.
    n_batch : int
        Number of batches per "epoch".

    Notes
    -----
    Based on code for sampling batches of PK images for triplet loss [1]_: 
    - https://github.com/VisualComputingInstitute/triplet-reid/blob/f3aed745964d81d7410e1ebe32eb4329af886d2d/train.py#L234-L250.

    If unique_concepts is not specified then y_labels can only be of type 
    array-like and not tf.Tensor.

    References
    ----------
    .. [1] A. Hermans, L. Beyer, B. Leibe (2017):
            In Defense of the Triplet Loss for Person Re-Identification.
            https://arxiv.org/abs/1703.07737
    
    Examples
    --------
    Create an iterator and get tensors for the batches of data and labels:

    >>> balanced_dataset, n_batches = ml.data.batch_k_examples_for_p_concepts(...)
    >>> balanced_dataset = balanced_dataset.prefetch(1)  # Parallel CPU/GPU processing
    >>> ...
    >>> with tf.Session() as sess:
    ...     ...
    ...     for epoch in range(n_epochs):
    ...         # Create new iterator and loop over balanced P.K dataset:
    ...         x_batch, y_batch = balanced_dataset.make_one_shot_iterator().get_next()
    ...         for i in range(n_batches):
    ...             sess.run([...], feed_dict={x_in: x_batch, y_in: y_batch})
    ...         # End of epoch
    """
    # # Get the unique concept labels (if None, y_labels can't be tensor)
    # if unique_concepts is None:
    #     if isinstance(y_labels, tf.Tensor):
    #         raise TypeError("Input for y_labels cannot be of type tf.Tensor if "
    #                         "unique_concept is not specified.")
    #     unique_concepts = np.unique(y_labels)   
    # n_concepts = np.shape(unique_concepts)[0]
    # n_batches = n_concepts // p_batch
    # n_dataset = n_batches * p_batch  # Multiple of P batch size

    # Get unique concept labels and count
    unique_concepts = tf.unique(y_labels)[0]
    n_concepts = tf.shape(unique_concepts, out_type=tf.int64)[0]
    # Create shuffled dataset of the unique concept labels
    balanced_dataset = tf.data.Dataset.from_tensor_slices(unique_concepts)
    balanced_dataset = balanced_dataset.shuffle(n_concepts)
    # Select p_batch labels from the shuffled concepts for one batch/episode
    balanced_dataset = balanced_dataset.take(p_batch) 
    # Map each of the selected concepts to a set of K exemplars
    balanced_dataset = balanced_dataset.flat_map(
        lambda concept: tf.data.Dataset.from_tensor_slices(
            _sample_k_examples_for_labels(labels=concept,
                                          x_data=x_data,
                                          y_labels=y_labels,
                                          k_size=k_batch)))
    # Group flattened dataset into batches of P.K exemplars
    balanced_dataset = balanced_dataset.batch(p_batch * k_batch)
    # Repeat dataset indefinitely (should be controlled by n_episodes)
    balanced_dataset = balanced_dataset.repeat(count=-1) 
    return balanced_dataset


# def _sample_k_examples_for_concept(
#             concept,
#             x_data,
#             y_labels,
#             k_batch):
#         """Sample K examples for a given concept label.
        
#         Based on:
#         - https://github.com/VisualComputingInstitute/triplet-reid/blob/f3aed745964d81d7410e1ebe32eb4329af886d2d/train.py#L145-L161
#         """
#         all_examples = tf.boolean_mask(x_data, tf.equal(y_labels, concept))
#         # TODO(rpeloff) repeat examples for size less than K examples
#         # Get example indices and shuffle 
#         indices = tf.range(tf.shape(all_examples)[0])
#         indices = tf.random_shuffle(indices)
#         # Select first K shuffled examples and return with concept label
#         k_examples = tf.gather(all_examples, indices[:k_batch])
#         return (k_examples,
#                 tf.fill([k_batch], concept))


def sample_dataset_triplets(
        dataset,
        use_dummy_data=False,
        x_fake_data=None,
        y_fake_data=None,
        n_max_same_pairs=int(100e3)):
    """Sample dataset of triplets from another dataset.

    Useful for sampling triplets in a tf.data.Dataset pipeline.
    
    Parameters
    ----------
    dataset : tf.data.Dataset
        A tf.data dataset containing batches of (x_data, y_labels).
    n_max_same_pairs : int, optional
        The maximum number of same pairs that may be sampled (the default is int(100e3)).
    
    Returns
    -------
    triplet_dataset : tf.data.Dataset
        Triplets dataset containing batches of (x_triplets, y_triplets).

    Notes
    -----
    The returned dataset consists of batches of (x_triplets, y_triplets), where:
    - x_triplets: tuple of triplet features (x_anchor, x_same, x_different).
    - y_triplets: tuple of triplet labels (y_anchor, y_same, y_different).

    See `sample_triplets` for more info.

    Examples
    --------
    Create a balanced batch dataset and use it to sample triplet batches:

    >>> balanced_dataset, n_batches = batch_k_examples_for_p_concepts(...)
    >>> triplet_dataset = batch_dataset_triplets(balanced_dataset)
    >>> ...
    >>> with tf.Session() as sess:
    ...     ...
    ...     for epoch in range(n_epochs):
    ...         # Create new iterator and loop over balanced P.K dataset:
    ...         x_triplets, y_triplets = triplet_dataset.make_one_shot_iterator().get_next()
    ...         for i in range(n_batches):
    ...             sess.run([...], feed_dict={x_in: x_triplets, y_in: y_triplets})
    ...         # End of epoch
    """
    triplet_dataset = dataset.map(
        lambda x_batch, y_batch: (
            sample_triplets(x_data=x_batch, 
                            y_labels=y_batch,
                            use_dummy_data=use_dummy_data,
                            x_fake_data=x_fake_data,
                            y_fake_data=y_fake_data,
                            n_max_same_pairs=n_max_same_pairs)))
    return triplet_dataset


def sample_triplets(
        x_data,
        y_labels,
        use_dummy_data=False,
        x_fake_data=None,
        y_fake_data=None,
        n_max_same_pairs=int(100e3)):
    """Sample all possible combinations of triplets from features and labels.

    Creates triplet data (x_triplet, y_triplet) containing all possible
    combinations of triplets in y_labels. A triplet consists of an "anchor",
    "same", and "different" observation, where the "anchor" and "same" 
    observations have the same concept class, and the "anchor" and "different"
    observations have different concept classes.

    For a balanced dataset of P concept classes, and K observations per class, 
    their will be P(K-1)(PK-K) total triplets.

    Parameters
    ----------
    x_data : array-like or tf.tensor
        Array or tensor of data features.
    y_labels : array-like or tf.tensor
        Array or tensor of data labels.
    n_max_same_pairs : int, optional
        The maximum number of same pairs that may be sampled (the default is int(100e3)).
    
    Returns
    -------
    triplets : nested tuple of tf.Tensor
        All possible combinations of triplets (x_triplets, y_triplets).

    Notes
    -----
    The returned tuple contains tensors (x_triplets, y_triplets), where:
    - x_triplets: tuple of triplet feature tensors (x_anchor, x_same, x_different).
    - y_triplets: tuple of triplet label tensors (y_anchor, y_same, y_different).
    """
    # Create 2D matches matrix where y_matches[i][j] is True if y[i] == y[j]
    y_matches = tf.equal(tf.expand_dims(y_labels, 1), y_labels)
    # Get indices [i,j], where y[i] is anchor, and y[j] is same/different label
    diff_pair_matches = tf.logical_not(y_matches)
    same_pair_matches = tf.linalg.set_diag(
        y_matches, tf.zeros(shape=tf.shape(y_labels)[0], dtype=tf.bool))
    same_pair_indices = tf.where(same_pair_matches)
    diff_pair_indices = tf.where(diff_pair_matches)
    # Get all possible triplet indices 
    triplet_indices, n_trip = _get_all_triplets_from_indices(same_pair_indices,
                                                             diff_pair_indices,
                                                             n_max_same_pairs)
    x_triplet = [tf.gather(x_data, triplet_indices[0]),
                 tf.gather(x_data, triplet_indices[1]),
                 tf.gather(x_data, triplet_indices[2])]
    y_triplet = [tf.gather(y_labels, triplet_indices[0]),
                 tf.gather(y_labels, triplet_indices[1]),
                 tf.gather(y_labels, triplet_indices[2])]
    # Use dummy data (if specified) when there are no triplets (n_triplets == 0)
    # Results in loss = margin since d(pos) = d(neg)
    if use_dummy_data:
        if x_fake_data is None:
            x_fake_data = tf.ones(shape=tf.shape(x_data)[1:], 
                                  dtype=tf.convert_to_tensor(x_data).dtype)
        if y_fake_data is None:
            y_fake_data = tf.constant(-1,
                                      dtype=tf.convert_to_tensor(y_labels).dtype)
        x_triplet = tf.cond(tf.equal(n_trip, 0),
            lambda: [x_fake_data, x_fake_data, x_fake_data],
            lambda: x_triplet)
        y_triplet = tf.cond(tf.equal(n_trip, 0),
            lambda: [y_fake_data, y_fake_data, y_fake_data],
            lambda: y_triplet)
    # Gather and return triplet data
    # x_triplet -> (x_anchor, x_same, x_different)
    # y_triplet -> (y_anchor, y_same, y_different)
    return [x_triplet, y_triplet]


def batch_few_shot_episodes(
        x_support_data,
        y_support_labels,
        x_query_data,
        y_query_labels,
        z_support_originators=None,
        z_query_originators=None,
        episode_label_set=None,
        make_matching_set=False,
        originator_type='different',
        k_shot=1,
        l_way=5,
        n_queries=10,
        seed=42):
    """Create datasets with batches of few-shot episodes.
    
    Used to create few-shot datasets (tf.data.Dataset) that may be iterated to 
    to form episodes of L-way support and query sets. Support set fetched for an
    episode contains K examples for each of the L-way episode labels, while the
    query set contains N query examples that correpsond with the episode labels.
    
    Parameters
    ----------
    x_support_data : array-like or tf.tensor
        Array or tensor of support data features.
    y_support_labels : array-like or tf.tensor
        Array or tensor of support data labels.
    x_query_data : array-like or tf.tensor
        Array or tensor of query data features.
    y_query_labels : array-like or tf.tensor
        Array or tensor of query data labels.
    z_support_originators : array-like or tf.tensor, optional
        Array or tensor of support data originators (e.g. speaker, drawer, etc.).
    z_query_originators : array-like or tf.tensor, optional
        Array or tensor of query data originators (e.g. speaker, drawer, etc.).
    k_shot : int, optional
        Number of support examples per concept in an episode (the default is 1, i.e. one-shot).
    l_way : int, optional
        Number of unique concepts/labels to sample per episode (the default is 5, i.e. 5-way).
    n_queries : int, optional
        Number of query examples to sample per episode (the default is 10).
    seed : int, optional
        Op-level seed for repeatable distributions across sessions (default is 42).
    
    Returns
    -------
    episode_data : tuple of tf.data.Dataset, (support_set, query_set)
        Support and query datasets which may be iteratated for l-way k-shot episodes.

    Notes
    -----
    This function makes use of the tf.random_shuffle method which creates
    stateful nodes in the computational graph. Therefore the support and query
    datasets require initializable iterators in order to initialize these nodes.

    The support and query features and labels may be generated as follows:
    >>> (x_support_data, x_query_data), (y_support_labels, y_query_labels) = (
    ...     make_train_test_split(
    ...         x_data, y_labels, test_ratio=0.5, shuffle=True, seed=42)
    
    Examples
    --------
    Generate 10 5-way episodes with 1-shot support sets and 3 queries:

    >>> support_set, query_set = batch_few_shot_episodes(
    ...     x_support_data,
    ...     y_support_labels,
    ...     x_query_data,
    ...     y_query_labels,
    ...     k_shot=1,
    ...     l_way=5,
    ...     n_queries=3
    ... )
    >>> iter_s = support_set.make_initializable_iterator()
    >>> iter_q = query_set.make_initializable_iterator()
    >>> with tf.Session() as sess:
    ...     sess.run([iter_s.initializer, iter_q.initializer])  # Init only once
    ...     for episode in range(10):
    ...         episode_data = sess.run([iter_s.get_next(), iter_q.get_next()])
    ...         x_support_batch, y_support_batch = episode_data[0]
    ...         x_query_batch, y_query_batch = episode_data[1]
    """
    # Create a dataset of episode labels which random samples L unique labels
    if episode_label_set is None:
        episode_label_set = create_episode_label_set(y_support_labels,
                                                     y_query_labels,
                                                     l_way=l_way,
                                                     seed=seed)
    # Create support set by sampling K examples for each of the episode labels
    support_set_size = k_shot * l_way
    support_set_temp = episode_label_set.flat_map(
        lambda label: tf.data.Dataset.from_tensor_slices(
            _sample_k_examples_for_labels(label,
                                          x_support_data,
                                          y_support_labels,
                                          z_originators=z_support_originators,
                                          # z_type='different'  # one of ['same', 'different', 'random']
                                          k_size=k_shot,
                                          seed=seed)))
    support_set_temp = support_set_temp.shuffle(support_set_size, seed=seed)
    support_set_temp = support_set_temp.batch(support_set_size)  # fetch entire support set
    support_set = support_set_temp.repeat(-1)  # repeat indefinitely (infinite episodes)
    # Create query set by sampling examples corresponding to episode labels in
    # support set, and with different originators to support set (if specified)       
    if z_support_originators is not None and z_query_originators is not None: 
        query_set = support_set_temp.flat_map(
            lambda x_support, y_support, z_originators: (  # y_support := episode label set
                tf.data.Dataset.from_tensor_slices(
                    _sample_k_examples_for_labels(
                        y_support,
                        x_query_data,
                        y_query_labels,
                        z_originators=z_query_originators,
                        valid_mask=_get_originator_mask(z_originators, 
                                                        z_query_originators,
                                                        originator_type=originator_type),
                        k_size=n_queries,
                        balanced=make_matching_set,  # NOTE n_queries should equal l_way for matching set
                        seed=seed))
            ))
    else:  # no originators specified
        query_set = support_set_temp.flat_map(
            lambda x_support, y_support: (  # y_support := episode label set
                    tf.data.Dataset.from_tensor_slices(
                        _sample_k_examples_for_labels(
                        y_support,
                        x_query_data,
                        y_query_labels,
                        k_size=n_queries,
                        balanced=make_matching_set,  # NOTE n_queries should equal l_way for matching set
                        seed=seed))
            ))
    query_set = query_set.shuffle(n_queries, seed=seed)
    query_set = query_set.batch(n_queries)  # fetch entire query set
    query_set = query_set.repeat(-1)  # repeat indefinitely (infinite episodes)
    # Return the support and query tf.data datasets
    return tf.data.Dataset.zip((support_set, query_set))


# def batch_mulitmodal_few_shot_episodes(
#         x1_support_data,
#         y1_support_labels,
#         x2_support_data,
#         y2_support_labels,
#         x_query_data,
#         y_query_labels,
#         x_matching_data,
#         y_matching_labels,
#         z1_support_originators=None,
#         z2_support_originators=None,
#         z_query_originators=None,
#         z_matching_originators=None,
#         k_shot=1,
#         l_way=5,
#         n_queries=10,
#         seed=42):
#     """Create datasets with batches of multimodal few-shot episodes.
    
#     See `batch_few_shot_episodes` above for more information. Same as unimodal
#     case except that support set examples consist of paired data from two 
#     modalities (based on correponding labels), and an L-way matching set with 
#     one example per unique episode label is returned along with the query set. 
    
#     For example, two support sets are returned which could contain the 
#     mulitmodal examples (x_image, y_label) and (x_speech, y_label) respectively,
#     where y_label from both sets correspond.
    
#     Parameters
#     ----------
#     x1_support_data : array-like or tf.tensor
#         Array or tensor of support data features for first modality.
#     y1_support_labels : array-like or tf.tensor
#         Array or tensor of support data labels for first modality.
#     x2_support_data : array-like or tf.tensor
#         Array or tensor of support data features for second modality.
#     y2_support_labels : array-like or tf.tensor
#         Array or tensor of support data labels for second modality.
#     x_query_data : array-like or tf.tensor
#         Array or tensor of query data features.
#     y_query_labels : array-like or tf.tensor
#         Array or tensor of query data labels.
#     x_matching_data : array-like or tf.tensor
#         Array or tensor of matching data features (disjoint or in different modality from query data).
#     y_matching_labels : array-like or tf.tensor
#         Array or tensor of matching data labels (disjoint or in different modality from query data).
#     k_shot : int, optional
#         Number of support examples per concept in an episode (the default is 1, i.e. one-shot).
#     l_way : int, optional
#         Number of unique concepts/labels to sample per episode (the default is 5, i.e. 5-way).
#     n_queries : int, optional
#         Number of query examples to sample per episode (the default is 10).
#     seed : int, optional
#         Op-level seed for repeatable distributions across sessions (default is 42).
    
#     Returns
#     -------
#     episode_data : tuple of tf.data.Dataset, (x1_support_set, x2_support_set, query_set, matching_set)
#         Multimodal support, query, and matching datasets which may be iteratated for l-way k-shot episodes.
#     """
#     # Create a dataset of episode labels which random samples L unique labels
#     episode_label_set = create_episode_label_set(y1_support_labels,
#                                                   y2_support_labels,
#                                                   y_query_labels,
#                                                   y_matching_labels,
#                                                   l_way=l_way,
#                                                   seed=seed)
#     # Create support dataset for first modality (K examples per episode label)
#     support_set_size = k_shot * l_way
#     x1_support_set = episode_label_set.flat_map(
#         lambda label: tf.data.Dataset.from_tensor_slices(
#             _sample_k_examples_for_labels(label,
#                                           x1_support_data,
#                                           y1_support_labels,
#                                           k_size=k_shot,
#                                           seed=seed)))
#     x1_support_set = x1_support_set.shuffle(support_set_size, seed=seed)
#     x1_support_set = x1_support_set.batch(support_set_size)  # fetch entire support set
#     x1_support_set = x1_support_set.repeat(-1)  # repeat indefinitely (infinite episodes)
#     # Similarly create paired support dataset for second modality
#     x2_support_set = episode_label_set.flat_map(
#         lambda label: tf.data.Dataset.from_tensor_slices(
#             _sample_k_examples_for_labels(label,
#                                           x2_support_data,
#                                           y2_support_labels,
#                                           k_size=k_shot,
#                                           seed=seed)))
#     x2_support_set = x2_support_set.shuffle(support_set_size, seed=seed)
#     x2_support_set = x2_support_set.batch(support_set_size)  # fetch entire support set
#     x2_support_set = x2_support_set.repeat(-1)  # repeat indefinitely (infinite episodes)
#     # Create query dataset by random sampling examples for the episode labels 
#     query_set = episode_label_set.batch(l_way).flat_map(
#         lambda label_batch: tf.data.Dataset.from_tensor_slices(
#             _sample_k_examples_for_labels(label_batch,
#                                           x_query_data,
#                                           y_query_labels,
#                                           k_size=n_queries,
#                                           seed=seed)))
#     query_set = query_set.shuffle(n_queries, seed=seed)
#     query_set = query_set.batch(n_queries)  # fetch entire query set
#     query_set = query_set.repeat(-1)  # repeat indefinitely (infinite episodes)
#     # Create matching dataset with one matching example per episode label
#     matching_set = episode_label_set.flat_map(
#         lambda label: tf.data.Dataset.from_tensor_slices(
#             _sample_k_examples_for_labels(label,
#                                           x_matching_data,
#                                           y_matching_labels,
#                                           k_size=1,
#                                           seed=seed)))
#     matching_set = matching_set.shuffle(l_way, seed=seed)
#     matching_set = matching_set.batch(l_way)  # fetch entire support set
#     matching_set = matching_set.repeat(-1)  # repeat indefinitely (infinite episodes)
#     # Return the multimodal support, query, and matching tf.data datasets
#     return x1_support_set, x2_support_set, query_set, matching_set


def create_episode_label_set(
        *y_label_sets,
        l_way=5,
        seed=42):
    """Create a dataset pipeline of L unique labels for a few-shot episode."""
    # Get valid label set as labels that occur in both support and qeury set
    if len(y_label_sets) < 1:
        raise ValueError("At least one label set required for y_label_sets.")
    valid_labels_set = tf.convert_to_tensor(y_label_sets[0])
    for label_set in y_label_sets[1:]:
        valid_labels_set = tf.sets.set_intersection(
            valid_labels_set[None, :],
            tf.convert_to_tensor(label_set)[None, :]).values
    labels_set_size = tf.cast(tf.shape(valid_labels_set)[0], tf.int64)
    # Create episode label set as L labels shuffled and taken from valid labels
    episode_label_set = tf.data.Dataset.from_tensor_slices(valid_labels_set)
    episode_label_set = episode_label_set.shuffle(labels_set_size, seed=seed)
    episode_label_set = episode_label_set.take(l_way)
    return episode_label_set



def _sample_k_examples_for_labels(
        labels,
        x_data,
        y_labels,
        z_originators=None,
        valid_mask=None,
        k_size=1,
        balanced=False,
        seed=42):
    """Sample k examples from the data within the specified label subset."""
    label_subset_mask = tf.equal(tf.expand_dims(y_labels, -1), labels)
    if valid_mask is not None:
        label_subset_mask = tf.cast(tf.reduce_sum(tf.cast(
            label_subset_mask, tf.int32), axis=-1), tf.bool)
        label_subset_mask = tf.logical_and(label_subset_mask, valid_mask)
    label_subset_indices = tf.where(label_subset_mask)[:, 0]
    
    y_label_subset = tf.gather(y_labels, label_subset_indices)
    x_label_subset = tf.gather(x_data, label_subset_indices)
        
    # Shuffle label subset examples
    shuffle_indices = tf.range(tf.shape(y_label_subset)[0])
    shuffle_indices = tf.random_shuffle(shuffle_indices, seed=seed)
    x_label_subset = tf.gather(x_label_subset, shuffle_indices)
    y_label_subset = tf.gather(y_label_subset, shuffle_indices)
    if z_originators is not None:
        z_label_subset = tf.gather(z_originators, label_subset_indices)
        z_label_subset = tf.gather(z_label_subset, shuffle_indices)
    
    if balanced:
        unique_labels = tf.unique(y_label_subset)[0]
        n_unique = tf.shape(unique_labels)[0]
        label_size = tf.cast(k_size/n_unique, tf.int32)
        valid_indices = tf.constant([], dtype=tf.int32)
        counter = tf.constant(0)
        
        x = [counter, n_unique, label_size, y_label_subset, unique_labels, valid_indices]
        
        def cond(i, n_unique, k_labels, labels, unique_labels, valid_indices):
            return tf.less(i, n_unique)
        
        def body(i, n_unique, k_labels, labels, unique_labels, valid_indices):
            label = tf.gather(unique_labels, i)
            label_subset_mask = tf.equal(tf.expand_dims(labels, -1), label)
            label_subset_indices = tf.where(label_subset_mask)[:, 0][:k_labels]
            valid_indices = tf.concat([valid_indices, tf.cast(label_subset_indices, tf.int32)], axis=0)
            return (i+1, n_unique, k_labels, labels, unique_labels, valid_indices)
        
        loop = tf.while_loop(cond, body, x, shape_invariants=[tf.TensorShape([]), tf.TensorShape([]),
            tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])])
        
        balanced_indices = loop[5]
        y_label_subset = tf.gather(y_label_subset, balanced_indices)
        x_label_subset = tf.gather(x_label_subset, balanced_indices)
        if z_originators is not None:
            z_label_subset = tf.gather(z_label_subset, balanced_indices)
            
    # Shuffle label subset examples and choose first k examples
#     k_shuffle_indices = tf.range(tf.shape(y_label_subset)[0])
#     k_shuffle_indices = tf.random_shuffle(k_shuffle_indices, seed=seed)[:k_size]
#     k_examples = tf.gather(x_label_subset, k_shuffle_indices)
#     k_labels = tf.gather(y_label_subset, k_shuffle_indices)
    # Choose first k examples
    k_examples = x_label_subset[:k_size]
    k_labels = y_label_subset[:k_size]
    if z_originators is not None:
        k_originators = z_label_subset[:k_size]
        return k_examples, k_labels, k_originators
    else:
        return k_examples, k_labels


def _get_all_triplets_from_indices(same_pair_indices, diff_pair_indices, n_max_same_pairs=int(100e3)):
    """Get all combinations of triplets from "same" and "different" pair indices."""
    anch_indices = same_pair_indices[:n_max_same_pairs, 0]  # "Anchor" observation indices
    same_indices = same_pair_indices[:n_max_same_pairs, 1]  # "Same" observation indices
    diff_indices = diff_pair_indices[:, 1]  # "Different" observation indices
    # "Anchor" observation indices for "Different" pairs
    diff_anch_indices = diff_pair_indices[:, 0]
    # Get triplet indices [i,j] of "different" and "same" observation indices, 
    # where same_pair_indices[i] and diff_pair_indices[j] have same "anchor"
    triplet_same_diff_indices = tf.where(tf.equal(
        tf.expand_dims(anch_indices, axis=1),
        diff_anch_indices))
    n_triplets = tf.shape(triplet_same_diff_indices)[0]
    # Gather the triplet observation indices
    anch_triplet_indices = tf.gather(anch_indices,
                                     triplet_same_diff_indices[:, 0])
    same_triplet_indices = tf.gather(same_indices,
                                     triplet_same_diff_indices[:, 0])
    diff_triplet_indices = tf.gather(diff_indices,
                                     triplet_same_diff_indices[:, 1])
    # Return tuple of (anchor, same, different) triplet indices
    return ((anch_triplet_indices,  # Anchor indices
             same_triplet_indices,  # Same indices
             diff_triplet_indices),  # Different indices
            n_triplets)            


def _get_originator_mask(
        z_support_set,
        z_query_originators,
        originator_type='different'):
    """Get a boolean mask over query dataset to select instances where originators are valid.
    
    originator_type one of ['different', 'same']:
    - 'different': Query and support set originators are always different
    - 'same': Query originator always appears in support set (not necessarily for matching concept)
    """
    valid_mask = tf.equal(
        tf.expand_dims(z_query_originators, -1), z_support_set)
    valid_mask = tf.reduce_sum(tf.cast(valid_mask, tf.int32), axis=-1)
    if originator_type == 'different':
        valid_mask = tf.equal(valid_mask, False)
    elif originator_type == 'same':
        valid_mask = tf.equal(valid_mask, True)
    return valid_mask
