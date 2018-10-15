"""Build functions for unimodal vision models.

Embedding Baselines:
- Pixels
- Feedforward Classifier
- Convolutional Classifier
- Siamese Triplets (Offline)
- Siamese Online Mining

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: August 2018
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging


import numpy as np
import tensorflow as tf


#pylint: disable=E0401
from mltoolset import neural_blocks
from mltoolset import siamese
from mltoolset import utils  
from mltoolset import TF_FLOAT, TF_INT
#pylint: enable=E0401


# Types of models that this module can build
MODEL_VERSIONS = [
    'pixels',
    'feedforward_softmax',
    'convolutional_softmax',
    'siamese_triplet',
    'siamese_triplet_online'
]


# Pure pixel matching model params
pixels_model_dict = {
    # --------------------
    # Other model options:
    # --------------------
    "resize_shape": [28, 28],  # define input (resize) shape
    "x_input_shape": [None, 28, 28, 1],  # define model input shape
}


# Base embedding model params (updated by model specific parameters)
base_model_dict = {
    # ----------------------
    # Convolutional options:
    # ----------------------
    "n_filters": None,  # NOTE: None specifies no convolutional layers
    "kernel_sizes": [], 
    "conv_strides": 1,  
    "conv_paddings": "valid",  # "valid" or "same"
    "pool_sizes": [],  
    "pool_strides": None,  # NOTE: None defaults to pool size
    "pool_paddings": "valid",  # "valid" or "same"
    "pool_type": "max",  # One of ["max", "avg", "avg_last"]
    "conv_activation": "relu",  # One of ["relu", "leaky_relu", "tanh", "sigmoid"]
    "conv_batch_norm": "before",  # One of [None, "before", "after"]
    "conv_bn_option_dict": {  # See tf.layers.batch_norm
            "fused": False  # default True leads to OOM with large tensors!
        }, 
    "dropout_channels": True,
    # --------------------
    # Feedforward options:
    # --------------------
    "n_hidden_units": None,  # NOTE: None specifies no hidden layers
    "n_linear_units": None,  # NOTE: None specifies no linear layers
    "hidden_activation": "relu",  # One of ["relu", "leaky_relu", "tanh", "sigmoid"]
    "hidden_batch_norm": "before",  # One of [None, "before", "after"]
    "hidden_bn_option_dict": {  # See tf.layers.batch_norm
            "fused": False  # default True leads to OOM with large tensors!
        },
    # ----------------------
    # General model options:
    # ----------------------
    "dropout_keep_prob": None,
    "normalize_output": False,
    "kernel_initializer": "glorot",  # One of ["zeros", "glorot", "glorot_normal", "pretorius"] 
    "bias_initializer": "zeros",  # One of ["zeros", "glorot", "glorot_normal"]
    "kernel_regularizer": None,  # (Only option for now)
    "bias_regularizer": None,  # (Only option for now)
    # ------------------------------
    # Siamese Triplet model options:
    # ------------------------------
    "triplet_margin": 0.2,
    "triplet_average_non_zero": False,
    # --------------------
    # Other model options:
    # --------------------
    "resize_shape": [28, 28],  # define input (resize) shape
    "x_input_shape": [None, 28, 28, 1],  # define model input shape
    "conv_input_shape": [28, 28, 1],  # define conv input shape
    "n_output_logits": None  # number of output classes (softmax)
}


# Feedforward embedding model params
ffnn_embedding_model_dict = base_model_dict.copy()
ffnn_embedding_model_dict.update({
    # --------------------
    # Feedforward options:
    # --------------------
    "n_hidden_units": [512, 512, 512],  # 3 hidden layers with 512 neurons each
    "n_linear_units": None,  # No linear output layer
    "hidden_batch_norm": "before",
})


# Convolutional embedding model params
cnn_embedding_model_dict = base_model_dict.copy()
cnn_embedding_model_dict.update({
    # ----------------------
    # Convolutional options:
    # ----------------------
    "n_filters": [32, 64, 128],  # Three layers with 32->64->128 output filters
    "kernel_sizes": [3, 3, 3],  # Three layers of (3x3) filters
    "conv_paddings": "valid",
    "pool_sizes": [2, 2, None], # (2x2) maxpool first two layers only  
    "pool_type": "max",
    "conv_batch_norm": "before",
    "dropout_channels": True, # dropout entire output filters (spatially)
    # --------------------
    # Feedforward options:
    # --------------------
    "n_hidden_units": [2048],  # 1 hidden layer with 2048 neurons
    "n_linear_units": None,  # No linear output layer
    "hidden_batch_norm": "before",
})


# Siamese triplet (online and offline) embedding model params
siamese_model_dict = cnn_embedding_model_dict.copy()
siamese_model_dict.update({
    # --------------------
    # Feedforward options:
    # --------------------
    "n_linear_units": 1024,  # 1024-dimensional linear output layer
    # ------------------------------
    # Siamese Triplet model options:
    # ------------------------------
    "triplet_margin": 0.2,  # Use margin of 0.2 between same/different pairs
    "triplet_average_non_zero": True,  # Only average non-zero triplet losses
    # ----------------------
    # General model options:
    # ----------------------
    "normalize_output": True
})


# Lookup dict for base model params
MODEL_BASE_PARAMS = {
    'pixels': pixels_model_dict,
    'feedforward_softmax': ffnn_embedding_model_dict,
    'convolutional_softmax': cnn_embedding_model_dict,
    'siamese_triplet': siamese_model_dict,
    'siamese_triplet_online': siamese_model_dict
}


def build_embedding_network(x_input,
                            **kwargs):
    """Build a base embedding model from an options dict."""
    # Sequentially build embedding model
    x_out = x_input

    # Build lower convolutional layers if n_filters specified
    if kwargs['n_filters'] is not None:
        x_out = neural_blocks.convolutional_layers(
            x_input=x_out,
            input_shape=kwargs['conv_input_shape'],
            n_layer_filters=kwargs['n_filters'],
            layer_kernel_sizes=kwargs['kernel_sizes'],
            layer_conv_strides=kwargs['conv_strides'],
            layer_conv_paddings=kwargs['conv_paddings'],
            layer_pool_sizes=kwargs['pool_sizes'],
            layer_pool_strides=kwargs['pool_strides'],
            layer_pool_paddings=kwargs['pool_paddings'],
            layer_pool_type=kwargs['pool_type'],
            activation=utils.literal_to_activation_func(
                kwargs['conv_activation']),
            kernel_initializer=utils.literal_to_initializer_func(
                kwargs['kernel_initializer'], kwargs['dropout_keep_prob']),
            bias_initializer=utils.literal_to_initializer_func(
                kwargs['bias_initializer']),
            kernel_regularizer=kwargs['kernel_regularizer'],
            bias_regularizer=kwargs['bias_regularizer'],
            keep_prob=kwargs['dropout_keep_prob'],
            drop_channels=kwargs['dropout_channels'],
            batch_norm=kwargs['conv_batch_norm'],
            bn_option_dict=kwargs['conv_bn_option_dict'],
            training=kwargs['training_flag']
        )
    # Build upper hidden dense layers if n_hidden_units specified
    if kwargs['n_hidden_units'] is not None:
        x_out = neural_blocks.hidden_layers(
            x_input=x_out,
            n_layer_units=kwargs['n_hidden_units'],
            activation=utils.literal_to_activation_func(
                kwargs['hidden_activation']),
            kernel_initializer=utils.literal_to_initializer_func(
                kwargs['kernel_initializer'], kwargs['dropout_keep_prob']),
            bias_initializer=utils.literal_to_initializer_func(
                kwargs['bias_initializer']),
            kernel_regularizer=kwargs['kernel_regularizer'],
            bias_regularizer=kwargs['bias_regularizer'],
            keep_prob=kwargs['dropout_keep_prob'],
            batch_norm=kwargs['hidden_batch_norm'],
            bn_option_dict=kwargs['hidden_bn_option_dict'],
            training=kwargs['training_flag']
        )
    # Build top linear layers if n_linear_units specified
    if kwargs['n_linear_units'] is not None:
        with tf.variable_scope('top_projection'):
            x_out = neural_blocks.dense_layer(
                x_input=x_out,
                n_units=kwargs['n_linear_units'],
                activation=None,
                kernel_initializer=utils.literal_to_initializer_func(
                    kwargs['kernel_initializer'], kwargs['dropout_keep_prob']),
                bias_initializer=utils.literal_to_initializer_func(
                    kwargs['bias_initializer']),
                kernel_regularizer=kwargs['kernel_regularizer'],
                bias_regularizer=kwargs['bias_regularizer'],
                name='linear'
            )
    # L2 normalize output embeddings if normalize_output is True
    if kwargs['normalize_output'] is True:
        x_out = tf.nn.l2_normalize(x_out, axis=1)
    # Return the embedding network
    return x_out


def softmax_classifier(x_input,
                       **kwargs):
    """Build a softmax classifier from an embedding model with output logits."""
    x_embed = build_embedding_network(x_input, **kwargs)

    if kwargs['n_output_logits'] is None:
        logging.info("Warning: Building softmax classifier without logits "
                     "layer. Value for n_output_logits is None. (This is fine "
                     "if only testing with the prior embedding layer).")
        return None, x_embed
    else:
        with tf.variable_scope('output_logits'):
            logits = neural_blocks.dense_layer(
                x_input=x_embed,
                n_units=kwargs['n_output_logits'],
                activation=None,
                kernel_initializer=utils.literal_to_initializer_func(
                    kwargs['kernel_initializer'], kwargs['dropout_keep_prob']),
                bias_initializer=utils.literal_to_initializer_func(
                    kwargs['bias_initializer']),
                kernel_regularizer=kwargs['kernel_regularizer'],
                bias_regularizer=kwargs['bias_regularizer'],
                name='logits'
            )
        return logits, x_embed


def build_vision_model(
        model_param_dict,
        x_train_data=None,
        y_train_labels=None,
        training=True):
    """Build the specified vision model, as well as loss func and metric ops.
    
    Returns (outputs, inputs, train_flag, loss_ops, metric_ops).
    """
    # Get model version
    model_version = model_param_dict['model_version']
    # Create boolean placeholder to indicate training or inference
    train_flag = tf.placeholder_with_default(False, shape=())
    model_param_dict = model_param_dict.copy()
    model_param_dict['training_flag'] = train_flag
    
    # Set loss func and metrics to None/empty if testing
    loss_func = None
    metrics = {}
    if model_version == 'pixels':
        # If model is pure pixel matching, only return embedding := flat input
        x_embed_input = tf.placeholder(TF_FLOAT, model_param_dict['x_input_shape'])
        embedding = tf.layers.flatten(x_embed_input)
    # Check model version and build corresponding model network
    elif (model_version == 'feedforward_softmax' or
          model_version == 'convolutional_softmax'):
        # Use input x_train_data for training if specified, otherwise
        # embedding placeholder input is used during inference
        if training and x_train_data is not None:
            x_embed_input = tf.cond(tf.equal(train_flag, True),
                                true_fn=lambda: x_train_data,
                                false_fn=lambda: tf.placeholder(
                                    TF_FLOAT, model_param_dict['x_input_shape']))
        else: 
            x_embed_input = tf.placeholder(TF_FLOAT,
                                           model_param_dict['x_input_shape'])
        # Build softmax classifier
        model_logits, embedding = softmax_classifier(x_embed_input, 
                                                     **model_param_dict)
        # Create train loss and metrics if training
        if training:
            # Define softmax loss with sparse labels (indices of one-hot encoding)
            loss_func = (
                tf.losses.sparse_softmax_cross_entropy(labels=y_train_labels,
                                                       logits=model_logits,
                                                       scope='loss'))
            # Define accuracy metric (as top K@1)
            with tf.name_scope('accuracy'):
                correct = tf.nn.in_top_k(predictions=model_logits,
                                        targets=y_train_labels,
                                        k=1)
                acc_metric = tf.reduce_mean(tf.cast(correct, TF_FLOAT))
                metrics = {"Accuracy": acc_metric}
    elif model_version == 'siamese_triplet':
        # Use anchor input x_train_data for training if specified, otherwise
        # embedding placeholder input is used during inference
        if training and x_train_data is not None:
            x_embed_input = tf.cond(tf.equal(train_flag, True),
                                true_fn=lambda: x_train_data[0],
                                false_fn=lambda: tf.placeholder(
                                    TF_FLOAT, model_param_dict['x_input_shape']))
        else: 
            x_embed_input = tf.placeholder(TF_FLOAT,
                                           model_param_dict['x_input_shape'])
        # Build siamese triplet network with shared params
        model_a, model_s, model_d = siamese.siamese_triplet_network(
            x_anchor_input=x_embed_input,
            x_same_input=x_train_data[1] if x_train_data is not None else x_embed_input,
            x_different_input=x_train_data[2] if x_train_data is not None else x_embed_input,
            model_func=build_embedding_network,
            model_param_dict=model_param_dict)
        embedding = model_a  # embedding layer is anchor model output
        # Define triplet loss function based on triplet pairs if training
        if training:
            loss_func = siamese.loss_triplets_cos(
                x_anchor=model_a,
                x_same=model_s,
                x_diff=model_d,
                margin=model_param_dict['triplet_margin'],
                average_non_zero=model_param_dict['triplet_average_non_zero'])
    elif model_version == 'siamese_triplet_online':
        # Use input x_train_data for training if specified, otherwise
        # embedding placeholder input is used during inference
        if training and x_train_data is not None:
            x_embed_input = tf.cond(tf.equal(train_flag, True),
                                true_fn=lambda: x_train_data,
                                false_fn=lambda: tf.placeholder(
                                    TF_FLOAT, model_param_dict['x_input_shape']))
        else: 
            x_embed_input = tf.placeholder(TF_FLOAT,
                                           model_param_dict['x_input_shape'])
        # Build online siamese triplet network (single embedding branch)
        embedding = build_embedding_network(x_embed_input, **model_param_dict)
        # Define online mining loss if training
        if training:
            loss_func = tf.contrib.losses.metric_learning.triplet_semihard_loss(
                labels=tf.reshape(y_train_labels, [-1]),
                embeddings=embedding,
                margin=model_param_dict['triplet_margin'])
    return (embedding,
            x_embed_input,
            train_flag,
            loss_func,
            metrics)
