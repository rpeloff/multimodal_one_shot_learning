"""Just some utility functions. :)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: August 2018
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import logging
import collections


import tensorflow as tf
import matplotlib.pyplot as plt


def set_logger(log_path, log_fn="train.log"):
    """Set logging to log info in terminal and a file `log_path`. 
    
    Useful to replace print("...") statements with logging.info("...") in order
    to store log information in a file for later viewing, as well as display it
    in the console. 

    Parameters
    ----------
    log_path : str
        Path to the directory that will store the log file.
    log_fn : str, optional
        Filename t (the default is "train.log").

    Notes
    -----
    Based on logging function from:
    - https://cs230-stanford.github.io/logging-hyperparams.html.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Logging to file "log_path/log_fn"
        log_file = os.path.join(log_path, log_fn)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        # Logging to console (allows us to replace `print` with `logging.info`)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def is_list_like(arg):
    """Determine whether `arg` is list-like ([], {}, ndarray, etc.), excluding strings."""
    return isinstance(arg, collections.Iterable) and not isinstance(arg, str)


def literal_to_float_list(arg_literal : str):
    """Evaluate a string literal (e.g. '[1,2,3]') and return list of floats."""
    try:
        result = eval(arg_literal)
    except SyntaxError:  # Does not get caught by argparse ...
        raise ValueError()
    
    if not isinstance(result, list):
        print("Invalid type for evaluated literal: {}. "
              "Expected type: {}.".format(type(result), list))
        raise TypeError()  # Gets caught by argparse so message is not shown
    
    return [float(i) for i in result]


def literal_to_optimizer_class(op_literal : str):
    """Convert specified optimizer literal to a tensor op class."""
    valid_options = ['sgd', 'momentum', 'adagrad', 'adadelta', 'adam']
    if op_literal not in valid_options: 
        raise ValueError("Specified optimizer not a valid option: {}."
                         "Expected one of: {}.".format(op_literal, valid_options))
    if op_literal == 'sgd':
        optimizer_class = tf.train.GradientDescentOptimizer
    elif op_literal == 'momentum':
        optimizer_class = tf.train.MomentumOptimizer
    elif op_literal == 'adagrad':
        optimizer_class = tf.train.AdagradOptimizer
    elif op_literal == 'adadelta':
        optimizer_class = tf.train.AdadeltaOptimizer
    elif op_literal == 'adam':
        optimizer_class = tf.train.AdamOptimizer
    return optimizer_class


def literal_to_activation_func(op_literal : str):
    """Convert specified activation literal to a tensor op function."""
    valid_options = [None, 'relu', 'leaky_relu', 'tanh', 'sigmoid']
    if op_literal not in valid_options: 
        raise ValueError("Specified activation not a valid option: {}."
                         "Expected one of: {}.".format(op_literal, valid_options))
    activation = None
    if op_literal == 'relu':
        activation = tf.nn.relu
    elif op_literal == 'leaky_relu':
        activation = tf.nn.leaky_relu
    elif op_literal == 'tanh':
        activation = tf.nn.tanh
    elif op_literal == 'sigmoid':
        activation = tf.nn.sigmoid
    return activation


def literal_to_initializer_func(op_literal : str, keep_prob=None):
    """Convert specified activation literal to a tensor op function."""
    valid_options = [None, 'zeros', 'glorot', 'glorot_normal', 'pretorius']
    if op_literal not in valid_options: 
        raise ValueError("Specified activation not a valid option: {}."
                         "Expected one of: {}.".format(op_literal, valid_options))
    initializer = None
    if op_literal == 'zeros':
        initializer = tf.zeros_initializer()
    elif op_literal == 'glorot':
        initializer = tf.glorot_uniform_initializer()
    elif op_literal == 'glorot_normal':
        initializer = tf.glorot_normal_initializer()
    elif op_literal == 'pretorius':
        initializer = tf.variance_scaling_initializer(
            scale=2*keep_prob,  # init based on dropout keep probability
            distribution='untruncated_normal')
    return initializer


def save_image(image, filename, cmap='gray'):
    """Save image to pdf with pyplot."""
    dirname = os.path.dirname(filename)
    dirname = '.' if dirname == '' else dirname
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(image, cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
