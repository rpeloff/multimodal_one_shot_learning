"""Building blocks for various Neural Network architectures.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: May 2018
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import itertools


import numpy as np
import tensorflow as tf


from . import _globals
from . import utils


# ==============================================================================
#                              FEEDFORWARD BLOCKS
# ------------------------------------------------------------------------------


def dense_layer(
        x_input,
        n_units,
        activation=None,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        name=None):
    # Get number of input features, and create layer kernel and bias variables
    n_inputs = int(x_input.get_shape()[1])
    kernel = tf.get_variable(
        name='kernel', shape=(n_inputs, n_units), dtype=_globals.TF_FLOAT,
        initializer=kernel_initializer, regularizer=kernel_regularizer)
    bias = tf.get_variable(
        name='bias', shape=(n_units), dtype=_globals.TF_FLOAT,
        initializer=bias_initializer, regularizer=bias_regularizer)
    # Return a tensor that computes linear/activated weighted input plus bias term
    x_linear = tf.matmul(x_input, kernel) + bias
    if activation is not None:
        x_out = activation(x_linear)
    x_out = x_linear
    return x_out if name is None else tf.identity(x_out, name=name)


def hidden_layers(
        x_input,
        n_layer_units,
        activation=None,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        keep_prob=None,
        batch_norm=None,
        bn_option_dict=None,
        training=False,
        layer_var_scope_prefix='hidden'):
    """
    TODO(rpeloff) batch norm one of [None, 'before', 'after']
    TODO(rpeloff) for batch_norm='after', see: https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
    """
    # Check if list of activations specified for each layer, else duplicate activation for layers
    layers = n_layer_units
    n_layers = np.shape(n_layer_units)[0]
    if utils.is_list_like(activation):
        n_activations = np.shape(activation)[0]
        if n_activations != n_layers: 
            raise ValueError(
                "List of activations does not match number of hidden layers. "  
                "Got {} activations, expected {}.".format(n_activations, n_layers))
        layers = zip(layers, activation)
    else:
        layers = itertools.zip_longest(layers, [], fillvalue=activation)
    # Check batch norm type is valid
    batch_norm_types = [None, 'before', 'after']
    if batch_norm not in batch_norm_types:
        raise ValueError(
            "Invalid batch norm type: {}. "
            "Expected one of: {}.".format(batch_norm, batch_norm_types))
    # Sequentially build each hidden layer upon the previous layer (initially the flattened input layer)
    x_out = tf.layers.flatten(x_input)
    for index, layer in enumerate(layers):
        n_units, activation_func = layer
        with tf.variable_scope('{}{}'.format(layer_var_scope_prefix, index)):
            # Create linear layer (activation postponed till after potential batch norm)
            x_out = dense_layer(x_input=x_out,
                                n_units=n_units,
                                activation=None,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                name='dense')
            # Batch norm before activation/non-linearity
            if batch_norm == 'before':
                x_out = tf.layers.batch_normalization(inputs=x_out,
                                                      training=training,
                                                      name='batch_norm',
                                                      **(bn_option_dict or {}))
            # Postponed activation/non-linearity
            x_out = tf.identity(activation_func(x_out), name='activation')
            # Batch norm after activation/non-linearity
            if batch_norm == 'after':
                x_out = tf.layers.batch_normalization(inputs=x_out,
                                                      training=training,
                                                      name='batch_norm',
                                                      **(bn_option_dict or {}))
            # Dropout layer if specified
            if keep_prob is not None:
                x_out = dropout_layer(x_input=x_out,
                                      keep_prob=keep_prob,
                                      training=training,
                                      name='dropout')
    return x_out


# ==============================================================================
#                              CONVOLUTIONAL BLOCKS
# ------------------------------------------------------------------------------


def conv2d_layer(
        x_input,
        n_filters,
        kernel_size,  # kernel_shape=(3,3) same as kernel_shape=3
        strides=(1, 1),  # same as strides=1
        padding='valid',
        activation=None,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        name=None):
    """
    TODO(rpeloff) kernel size := (filter_height, filter_width) OR (kernel_size, kernel_size)
    TODO(rpeloff) strides := (stride_height, stride_width) OR (strides, strides)
    TODO(rpeloff) data format assumed to be "channels last" (i.e. [n_batch, height, width, in_channels])
    """
    # Check the padding type is valid
    padding = padding.upper()
    padding_types = ['VALID', 'SAME']
    if padding.upper() not in padding_types: 
        raise ValueError(
            "Invalid padding type (case-insensitive): {}. "
            "Expected one of: {}.".format(padding, padding_types))
    # Get convolution window shape based on kernel_size and number of input and output channels/filters
    in_channels = int(x_input.get_shape()[-1])
    kernel_shape = (kernel_size, kernel_size) if np.shape(kernel_size) == () else tuple(kernel_size)
    kernel_shape += (in_channels, n_filters)  # := [height, width, in_channels, out_channels]
    # Create layer kernel (filters) and bias variables
    kernel = tf.get_variable(
        name='kernel', shape=kernel_shape, dtype=_globals.TF_FLOAT,
        initializer=kernel_initializer, regularizer=kernel_regularizer)
    bias = tf.get_variable(
        name='bias', shape=(n_filters), dtype=_globals.TF_FLOAT,
        initializer=bias_initializer, regularizer=bias_regularizer)
    # Get stride shape from strides scalar or tuple
    strides_shape = [1,] + ([strides, strides] if np.shape(strides) == () else list(strides)) + [1,]
    # Return a tensor that computes linear/activated 2D convolution on the input plus a bias term
    x_conv = tf.nn.conv2d(input=x_input,
                          filter=kernel,
                          strides=strides_shape,
                          padding=padding,
                          data_format='NHWC')
    x_conv = tf.nn.bias_add(x_conv, bias)
    if activation is not None:
        x_out = activation(x_conv)
    x_out = x_conv
    return x_out if name is None else tf.identity(x_out, name=name)


def _pooling2d_layer(
        x_input,
        pooling_func,
        pool_size,
        strides=None,
        padding='valid',
        name=None):
    """
    TODO(rpeloff) data format assumed to be "channels last" (i.e. [n_batch, height, width, in_channels])
    """
    # Check the padding type is valid
    padding = padding.upper()
    padding_types = ['VALID', 'SAME']
    if padding.upper() not in padding_types: 
        raise ValueError(
            "Invalid padding type (case-insensitive): {}. "
            "Expected one of: {}.".format(padding, padding_types))
    # Get pooling window shape from pool_size scalar or tuple
    pool_shape = [1,] + ([pool_size, pool_size] if np.shape(pool_size) == () else list(pool_size)) + [1,]
    # Get stride shape from strides scalar or tuple, or pool_size if None
    if strides is None:
        strides = pool_size
    strides_shape = [1,] + ([strides, strides] if np.shape(strides) == () else list(strides)) + [1,]
    # Return a tensor that computes pooling2d func on the input
    return pooling_func(value=x_input,
                        ksize=pool_shape,
                        strides=strides_shape,
                        padding=padding,
                        data_format='NHWC',
                        name=name)


def max_pooling2d_layer(
        x_input,
        pool_size,
        strides=None,
        padding='valid',
        name=None):
    """
    TODO(rpeloff) data format assumed to be "channels last" (i.e. [n_batch, height, width, in_channels])
    TODO(rpeloff) stride=None uses pool_size as strides
    """
    # Return a tensor that computes 2D max pooling on the input
    return _pooling2d_layer(x_input=x_input,
                            pooling_func=tf.nn.max_pool,
                            pool_size=pool_size,
                            strides=strides,
                            padding=padding,
                            name=name)


def avg_pooling2d_layer(
        x_input,
        pool_size,
        strides=None,
        padding='valid',
        name=None):
    """
    TODO(rpeloff) data format assumed to be "channels last" (i.e. [n_batch, height, width, in_channels])
    TODO(rpeloff) stride=None uses pool_size as strides
    """
    # Return a tensor that computes 2D max pooling on the input
    return _pooling2d_layer(x_input=x_input,
                            pooling_func=tf.nn.avg_pool,
                            pool_size=pool_size,
                            strides=strides,
                            padding=padding,
                            name=name)


def global_pooling2d_layer():
    # TODO(rpeloff) max/Avg. Pooling across entire channel to produce output of shape [1, 1, channels_in]
    # TODO(rpeloff) makes it easier to test different input padding lengths without calculating
    #               filter size for final pool layer over remaining units every time
    pass


def convolutional_layers(
        x_input,
        input_shape,  # tuple of (height, width, channels), e.g. (128, 128, 3) for 128x128 RGB images
        # conv layers
        n_layer_filters,  # list of integers specifying number of output filters after each conv layer
        layer_kernel_sizes,  # list of integers/tuples specifying conv kernel size of cnn layer
        layer_conv_strides=(1, 1),  # integer/tuple, or list of integers/tuples 
        layer_conv_paddings='valid',  # string or list of strings specifying conv padding of each cnn layer
        # pooling
        layer_pool_sizes=None,  # list of integers/tuples specifying pool size of each cnn layer (None specifies no pooling in the cnn layers)
        layer_pool_strides=None,  # integer/tuple, or list of integers/tuples (None defaults to use layer_pool_size) 
        layer_pool_paddings='valid',  # string or list of strings specifying pool padding of each cnn layer
        layer_pool_type='max',  # String, one of ['max', 'avg', 'avg_last'], specifying pool type in the cnn layers
        # general
        activation=None,  # callable, or list of callables specifying activation after conv in each cnn layer
        kernel_initializer=None,  # None specifies tf.get_variable default initializer (Likely tf.glorot_uniform_initializer)
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None, 
        # regularization
        keep_prob=None,  # float, include dropout layers after cnn layers with specified keep probability (None specifies no dropout layers)
        drop_channels=None,  # boolean, specifies whether to dropout spatially (i.e. dropout entire channels) instead of standard dropout
                             # NOTE: None & False same, default None so reader is not confused since dropout must be activated by specifying keep_prob
        batch_norm=None,  # One of [None, 'before', 'after'], specifies whether to use batch norm before or after the activation function in cnn layers
        bn_option_dict=None,  # Additional keyword arg dict of options for tf.layers.batch_normalization (see fused option!)
        training=False,  # boolean or tf.placeholder to control whether the graph is being used during training or inference
        debug=False,
        layer_var_scope_prefix='cnn_layer'):
    """
    TODO(rpeloff) for batch_norm='after', see: https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
    
    TODO(rpeloff) example usage for MNIST simulation:
    >>> import mltoolset as ml
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> n_data = 5
    >>> n_inputs = 28*28
    >>> data = np.asarray(np.random.rand(n_data, n_inputs), dtype=NP_DTYPE)
    >>> x_input = tf.placeholder(ml.TF_FLOAT, [None, n_inputs])
    >>> input_shape = [28, 28, 1]
    >>> n_layer_filters = [32, 32, 64, 64, 128]
    >>> filter_sizes = [
            # 1st cnn stack (can specify as tuple) 
            (3, 3),  # filter shape of first layer:  32 filters with shape 3x3x1  =   288 weights
            (3, 3),  # filter shape of second layer: 32 filters with shape 3x3x32 = 9,216 weights
            # 2nd cnn stack (can specify as integers) 
            3,       # filter shape of third layer:  64 filters with shape 3x3x32 = 18,432 weights
            3,       # filter shape of fourth layer: 64 filters with shape 3x3x64 = 36,864 weights
            # 3rd cnn stack (can specify as list) 
            [3, 3]   # filter shape of fifth layer:  128 filters with shape 3x3x64 = 73,728 weights
        ]
    >>> conv_paddings = 'same'  # Output width/height same as input
    >>> pool_sizes = [
            # 1st cnn stack
            None,    # skip pooling in first layer
            (2, 2),  # pool shape of second layer: (2x2)
            # 2nd cnn stack (POOL layer at the end)
            None,    # skip pooling in third layer
            2,       # pool shape of fourth layer: (2x2)
            # 3rd cnn stack (no pooling)
            None     # skip pooling in fifth layer
        ]
    >>> activation = tf.nn.relu
    >>> keep_prob = 0.5
    >>> drop_channels = True
    >>> batch_norm = 'before'
    >>> training = True
    >>> debug = True
    >>> x_cnn = ml.neural_blocks.convolutional_layers(
                x_input=x_input,
                input_shape=input_shape,
                n_layer_filters=n_layer_filters,
                layer_kernel_sizes=filter_sizes,
                layer_conv_paddings=conv_paddings,
                layer_pool_sizes=pool_sizes,
                activation=activation,
                keep_prob=keep_prob,
                drop_channels=drop_channels,
                batch_norm=batch_norm,
                training=training,
                debug=debug     
            )
    """
    # Get the number of cnn layers
    n_layers = np.shape(n_layer_filters)[0]
    # Parse the conv2d layer properties
    conv2d_layer_properties = _parse_conv_property_list(
        n_layers=n_layers,
        layer_sizes=layer_kernel_sizes,
        layer_strides=layer_conv_strides,
        layer_paddings=layer_conv_paddings,
        name='conv2d')      
    # Create cnn layer properties list from layer filters and conv2d layers
    cnn_layer_properties = [n_layer_filters, conv2d_layer_properties]
    # Check if using pooling in cnn and add to cnn layer properties
    # Else duplicate `None` as each layers pool properties
    if layer_pool_sizes is not None:
        pool2d_layer_properties = _parse_conv_property_list(
            n_layers=n_layers,
            layer_sizes=layer_pool_sizes,
            layer_strides=layer_pool_strides,
            layer_paddings=layer_pool_paddings,
            name='pooling2d')
        cnn_layer_properties.append(pool2d_layer_properties)
    else:
        cnn_layer_properties.append([None for i in range(n_layers)])
    # Check activations is a list: add to cnn layer properties
    # Else duplicate for each layer property (single string)
    if utils.is_list_like(activation):
        n_activations = np.shape(activation)[0]
        if n_activations != n_layers: 
            raise ValueError(
                "List of activations does not match number of cnn layers. "  
                "Got list size {}, expected {}.".format(n_activations, n_layers))
        cnn_layer_properties.append(activation)
    else:
        cnn_layer_properties.append([activation for i in range(n_layers)])
    # Check pooling type is valid
    pool_types = ['max', 'avg', 'avg_last']
    if layer_pool_type not in pool_types:
        raise ValueError(
            "Invalid pooling2d type: {}. "
            "Expected one of: {}.".format(layer_pool_type, pool_types))
    # Check batch norm type is valid
    batch_norm_types = [None, 'before', 'after']
    if batch_norm not in batch_norm_types:
        raise ValueError(
            "Invalid batch norm type: {}. "
            "Expected one of: {}.".format(batch_norm, batch_norm_types))
    # Reshape the inputs to 4D tensor with shape [-1, height, width, channels]
    input_shape = [-1,] + input_shape  # Note: n_batch set to -1
    x_out = tf.reshape(x_input, input_shape)
    # Sequentially build each cnn layer upon the previous layer
    for index, layer in enumerate(zip(*cnn_layer_properties)):
        n_filters, conv_layer, pool_layer, activation_func = layer
        with tf.variable_scope('{}{}'.format(layer_var_scope_prefix, index)):
            # Create conv2d layer (activation postponed till after potential batch norm)
            conv_kernel_size, conv_strides, conv_padding = conv_layer
            x_out = conv2d_layer(x_input=x_out,
                                 n_filters=n_filters,
                                 kernel_size=conv_kernel_size,
                                 strides=conv_strides,
                                 padding=conv_padding,
                                 activation=None,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 name='conv2d')
            # Batch norm before activation/non-linearity
            if batch_norm == 'before':
                x_out = tf.layers.batch_normalization(inputs=x_out,
                                                      training=training,
                                                      name='batch_norm',
                                                      **(bn_option_dict or {}))
            # Postponed activation/non-linearity
            x_out = tf.identity(activation_func(x_out), name='activation')
            # Batch norm after activation/non-linearity
            if batch_norm == 'after':
                x_out = tf.layers.batch_normalization(inputs=x_out,
                                                      training=training,
                                                      name='batch_norm',
                                                      **(bn_option_dict or {}))
            # Dropout layer if specified (& dropout spatially if specified)
            if keep_prob is not None:
                dropout_func = dropout_layer
                if drop_channels:
                    dropout_func = dropout_channel_layer    
                x_out = dropout_func(x_input=x_out,
                                     keep_prob=keep_prob,
                                     training=training,
                                     name='dropout')
            # Add optional pooling2d layer after conv2d layer (if specified)
            pool_size, pool_strides, pool_padding = pool_layer
            if pool_layer is not None and pool_size is not None:
                pooling2d_func = max_pooling2d_layer
                pooling2d_name = 'max'
                if layer_pool_type == 'avg':
                    pooling2d_func = avg_pooling2d_layer
                    pooling2d_name = 'avg'
                elif layer_pool_type == 'avg_last' and index+1 == n_layers:
                    pooling2d_func = avg_pooling2d_layer
                    pooling2d_name = 'avg'
                x_out = pooling2d_func(x_input=x_out,
                                       pool_size=pool_size,
                                       strides=pool_strides,
                                       padding=pool_padding,
                                       name='{}_pool2d'.format(pooling2d_name))
    return x_out
        
         
def _parse_conv_property_list(
        n_layers,
        layer_sizes,
        layer_strides,
        layer_paddings,
        name='conv2d'):
    """Parse parameters for conv/pool cnn layers and return layer properties."""
    # Create list of conv (or pool) properties
    # Poperties (e.g. strides) have a value for each of `n_layers` in the cnn
    layer_properties = [layer_sizes]
    # Check that shape of layer sizes is the same as number of cnn layers
    n_layer_sizes = np.shape(layer_sizes)[0]
    if n_layer_sizes != n_layers:
        raise ValueError(
            "List of {} layer sizes does not match number of cnn layers. "  
            "Got list size {}, expected {}.".format(name, n_layer_sizes, n_layers))
    # Check strides is a list: add to layer properties (if shape is correct) 
    # Else duplicate for each layer property (single int or tuple)
    if isinstance(layer_strides, list): 
        n_strides = np.shape(layer_strides)[0]
        if n_strides != n_layers:
            raise ValueError(
                "List of {} layer strides does not match number of cnn layers. "  
                "Got list size {}, expected {}.".format(name, n_strides, n_layers))
        layer_properties.append(layer_strides)
    else:
        layer_properties.append([layer_strides for i in range(n_layers)])
    # Check padding is a list: add to layer properties (if shape is correct)
    # Else duplicate for each layer property (single string)
    if utils.is_list_like(layer_paddings):
        n_paddings = np.shape(layer_paddings)[0]
        if n_paddings != n_layers: 
            raise ValueError(
                "List of {} layer paddings does not match number of cnn layers. "  
                "Got list size {}, expected {}.".format(name, n_paddings, n_layers))
        layer_properties.append(layer_paddings)
    else:
        layer_properties.append([layer_paddings for i in range(n_layers)])
    # Zip layer properties and return as a list
    return [layer_prop for layer_prop in zip(*layer_properties)]


# ==============================================================================
#                              REGULARIZATION BLOCKS
# ------------------------------------------------------------------------------


def dropout_layer(
        x_input,
        keep_prob,
        training=False,  # NOTE: Recommend using tf.placeholder_with_default(False, shape=())
        noise_shape=None,
        name=None):
    # Create callables to return the inference (1.0) or train (`keep_prob`) dropout keep prob
    inference_keep_prob = lambda: tf.constant(1.0, dtype=_globals.TF_FLOAT)
    train_keep_prob = lambda: keep_prob
    # Return the train dropout keep prob if `training` is true, else inference dropout keep prob
    cond_keep_prob = tf.cond(tf.equal(training, tf.constant(True)),
                             true_fn=train_keep_prob,
                             false_fn=inference_keep_prob)
    # Create a dropout layer with the conditioned dropout keep probability
    return tf.nn.dropout(x=x_input,
                         keep_prob=cond_keep_prob,
                         noise_shape=noise_shape,
                         name=name)


def dropout_channel_layer(
        x_input,
        keep_prob,
        training=False,
        name=None):
    """
    TODO(rpeloff) data format of x_input assumed to be "channels last" (i.e. [n_batch, height, width, in_channels])
    TODO(rpeloff) drop entire feature maps (i.e. conv channels out) instead of parts of individual feature maps.
    
    TODO(rpeloff) TL;DR: May improve performance where training set size is small.

    From https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/noise.py#L111-L142

    Convenience function to drop full channels of feature maps.
    Adds a dropout layer that sets feature map channels to zero, across
    all locations, with probability p. For convolutional neural networks, this
    may give better results than independent dropout [1]_.

    References
    ----------
    .. [1] J. Tompson, R. Goroshin, A. Jain, Y. LeCun, C. Bregler (2014):
           Efficient Object Localization Using Convolutional Networks.
           https://arxiv.org/abs/1411.4280
    """
    # Set shape for dropout keep/drop mask such that batch and channel dimensions
    # are kept independently, while row and column dimensions will be kept/dropped
    noise_shape = [tf.shape(x_input)[0], 1, 1, tf.shape(x_input)[3]]
    # Create a dropout layer with noise shape specified to drop feature maps
    return dropout_layer(x_input=x_input,
                         keep_prob=keep_prob,
                         training=training,
                         noise_shape=noise_shape,
                         name=name)
