"""Testing script for multimodal models.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: August 2018
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import json
import logging
import argparse
import datetime


import matplotlib
matplotlib.use('Agg')  # use anitgrain rendering engine backend for non-GUI
from sklearn import preprocessing
import numpy as np
import tensorflow as tf


# Add upper-level 'src' directory to application path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


#pylint: disable=E0401
from dtw import speech_dtw
from mltoolset import data
from mltoolset import nearest_neighbour
from mltoolset import utils
from mltoolset import TF_FLOAT, TF_INT
#pylint: enable=E0401
import vision
import speech


# Static names for stored log and param files
MODEL_PARAMS_STORE_FN = 'model_params.json'
LOG_FILENAME = 'test_multimodal.log'


# Text 'labels' for digits in MNIST and TIDigits 
# NOTE 'z' split into 'z' (zero) and 'o' (oh) for option `--zeros-different`
DIGITS_LABELS = [
    'z', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def check_arguments():
    """Check command line arguments for `python test_speech.py`."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])

    # --------------------------------------------------------------------------
    # General script options (model type, storage, etc.):
    # --------------------------------------------------------------------------
    parser.add_argument('--speech-model-dir', 
                        type=os.path.abspath,
                        help="Path to a trained speech model directory",
                        required=True)
    parser.add_argument('--vision-model-dir', 
                        type=os.path.abspath,
                        help="Path to a trained vision model directory",
                        required=True)
    parser.add_argument('--output-dir', 
                        type=os.path.abspath,
                        help="Path to store multimodal test results "
                             "(defaults to the current working directory)",
                        default='.')
    parser.add_argument('--speech-data-dir', 
                        type=os.path.abspath,
                        help="Path to store and load speech data"
                             "(defaults to '{}' in the current working "
                             "directory)".format('data'),
                        default='data')
    parser.add_argument('--vision-data-dir', 
                        type=os.path.abspath,
                        help="Path to store and load vision data"
                             "(defaults to '{}' in the current working "
                             "directory)".format('data'),
                        default='data')
    parser.add_argument('--params-file', 
                        type=str,
                        help="Filename of model parameters file e.g. '{0}' "
                             "(defaults to '{0}' in model directory "
                             "if found, else base model parameters "
                             "are used)".format(MODEL_PARAMS_STORE_FN),
                        default=None)
    parser.add_argument('--speech-restore-checkpoint', 
                        type=str,
                        help="Filename of speech model checkpoint to restore and "
                             "test (defaults to None which uses best model)",
                        default=None)
    parser.add_argument('--vision-restore-checkpoint', 
                        type=str,
                        help="Filename of vision model checkpoint to restore and "
                             "test (defaults to None which uses best model)",
                        default=None)
    parser.add_argument('-rs', '--random-seed', 
                        type=int,
                        help="Random seed (default: 42)",
                        default=42)
    
    # --------------------------------------------------------------------------
    # Model and data pipeline options:
    # --------------------------------------------------------------------------
    parser.add_argument('--test-set',
                        type=str,
                        help="Dataset to use for few-shot multimodal testing "
                             "(defaults to '{}', i.e. MNIST and TIDigits)"
                             "".format('digits'),
                        choices=['digits', 'flickr'],
                        default='digits')
    parser.add_argument('--query-type',
                    type=str,
                    help="Define which modality to use as query"
                         "(defaults to '{}', and matching set as '{}')"
                         "".format('speech', 'images'),
                    choices=['speech', 'images'],
                    default='speech')
    parser.add_argument('--zeros-different',
                       action='store_true',
                       help="Treat speech digits 'oh' and 'zero' as different "
                            "classes (default treats them as the same class)")
    parser.add_argument('--l-way',
                        type=int,
                        help="Number of L unique classes used in few-shot "
                             "evaluation (defaults to {})".format(10),
                        default=10)
    parser.add_argument('--k-shot',
                        type=int,
                        help="Number of K examples sampled per L-way label "
                             "used in few-shot evaluation (defaults to {})"
                             "".format(1),
                        default=1)
    parser.add_argument('--n-queries',
                        type=int,
                        help="Number of N_queries query examples to sample per "
                             "few-shot episode (defaults to {})".format(10),
                        default=10)
    parser.add_argument('--n-test-episodes',
                        type=int,
                        help="Number of few-shot test episodes"
                             "(defaults to {})".format(400),
                        default=400)
    
    return parser.parse_args()


def main():
    # --------------------------------------------------------------------------
    # Parse script args and handle options:
    # --------------------------------------------------------------------------
    ARGS = check_arguments()
    
    # Set numpy and tenorflow random seed
    np.random.seed(ARGS.random_seed)
    tf.set_random_seed(ARGS.random_seed)

    # Get specified model and output directories
    speech_model_dir = ARGS.speech_model_dir
    vision_model_dir = ARGS.vision_model_dir
    test_model_dir = ARGS.output_dir
    
    # Check if not using a previous run, and create a unique run directory
    if not os.path.exists(os.path.join(test_model_dir, LOG_FILENAME)):
        unique_dir = "{}_{}".format(
            'multimodal', 
            datetime.datetime.now().strftime("%y%m%d_%Hh%Mm%Ss_%f"))
        test_model_dir = os.path.join(test_model_dir, unique_dir)

    # Create directories
    if not os.path.exists(test_model_dir):
        os.makedirs(test_model_dir)
    
    # Set logging to print to console and log to file
    utils.set_logger(test_model_dir, log_fn=LOG_FILENAME)
    logging.info("Using vision model directory: {}".format(vision_model_dir))
    logging.info("Using speech model directory: {}".format(speech_model_dir))
    
    # Load JSON model params
    speech_model_params = load_model_params(speech_model_dir, ARGS.params_file,
        modality='speech')
    vision_model_params = load_model_params(vision_model_dir, ARGS.params_file,
        modality='vision')
    if speech_model_params is None or vision_model_params is None:
        return  # exit ...

    # Read and write testing options from specified/default args
    test_options = {}
    var_args = vars(ARGS)
    for arg in var_args:
        test_options[arg] = getattr(ARGS, arg)
    logging.info("Testing parameters: {}".format(test_options))
    test_options_path = os.path.join(test_model_dir, 'test_options.json')
    with open(test_options_path, 'w') as fp:
        logging.info("Writing most recent testing parameters to file: {}"
                        "".format(test_options_path))
        json.dump(test_options, fp, indent=4)
    
    # --------------------------------------------------------------------------
    # Get additional model parameters:
    # --------------------------------------------------------------------------
    feats_type = speech_model_params['feats_type']
    n_padding = speech_model_params['n_padded']
    center_padded = speech_model_params['center_padded']
    n_filters = 39 if (feats_type == 'mfcc') else 40
    image_size = 28 if (ARGS.test_set == 'digits') else None  # TODO(rpeloff)

    if n_padding is None or speech_model_params['model_version'] == 'dtw':
        n_padding = 110  # pad to longest segment length in TIDigits (DTW)
        center_padded = False
    
    # --------------------------------------------------------------------------
    # Load test datasets:
    # --------------------------------------------------------------------------
    if ARGS.test_set == 'digits':  # load digits (default) test set
        # Load MNIST data arrays
        logging.info("Testing vision model on dataset: {}".format('mnist'))
        vision_test_data = data.load_mnist()
        vision_inverse_data = False  # don't inverse mnist grayscale
        
        logging.info("Testing speech model on dataset: {}".format('tidigits'))
        tidigits_data = data.load_tidigits(
            path=os.path.join(ARGS.speech_data_dir, 'tidigits_audio.npz'),
            feats_type=feats_type)
        speech_test_data = tidigits_data[2]
    else:  # load flickr test set TODO(rpeloff)
        raise NotImplementedError()
        
    # --------------------------------------------------------------------------
    # Data processing pipeline (placed on CPU so GPU is free):
    # --------------------------------------------------------------------------
    with tf.device('/cpu:0'):
        # ---------------------------------------------
        # Create speech few-shot test dataset pipeline:
        # ---------------------------------------------
        x_speech_test = speech_test_data[0]
        y_speech_test = speech_test_data[1]
        z_speech_test = speech_test_data[2]
        x_speech_test_placeholder = tf.placeholder(TF_FLOAT, 
            shape=[None, n_filters, n_padding])
        y_speech_test_placeholder = tf.placeholder(tf.string, shape=[None])
        z_speech_test_placeholder = tf.placeholder(tf.string, shape=[None])
        # Preprocess speech data and labels
        x_speech_test = data.pad_sequences(x_speech_test,
                                           n_padding,
                                           center_padded=center_padded)
        x_speech_test = np.swapaxes(x_speech_test, 2, 1)  # (n_filters, n_pad)
        if not ARGS.zeros_different:  # treat 'oh' and 'zero' as same class
            y_speech_test = [
                word if word != 'o' else 'z' for word in y_speech_test]
        # Add single depth channel to feature image so it is a 'grayscale image'
        x_speech_test_with_depth= tf.expand_dims(x_speech_test_placeholder,
                                                 axis=-1)
        # Split data into disjoint support and query sets
        x_speech_test_split, y_speech_test_split, z_speech_test_split = (
            data.make_train_test_split(x_speech_test_with_depth,
                                       y_speech_test_placeholder,
                                       z_speech_test_placeholder,
                                       test_ratio=0.5,
                                       shuffle=True,
                                       seed=ARGS.random_seed))
        
        # ---------------------------------------------
        # Create vision few-shot test dataset pipeline:
        # ---------------------------------------------
        x_vision_test = vision_test_data[1][0]
        y_vision_test = vision_test_data[1][1]
        x_vision_test_placeholder = tf.placeholder(TF_FLOAT, 
            shape=[None, image_size, image_size])
        y_vision_test_placeholder = tf.placeholder(tf.string, shape=[None])
        # Preprocess image data and labels
        x_vision_test_preprocess = (
            data.preprocess_images(images=x_vision_test_placeholder,
                                   normalize=True,
                                   inverse_gray=vision_inverse_data,  
                                   resize_shape=vision_model_params['resize_shape'],
                                   resize_method=tf.image.ResizeMethod.BILINEAR,
                                   expand_dims=True,
                                   dtype=TF_FLOAT))
        y_vision_test = np.array([  # convert MNIST classes to TIDigits labels
            DIGITS_LABELS[digit] for digit in y_vision_test], dtype=str)
        if ARGS.zeros_different:  # treat 'oh' and 'zero' as different classes
            zero_ind = np.where(np.isin(y_vision_test, 'z'))[0]
            oh_ind = np.random.choice(zero_ind, 
                                      size=int(zero_ind.shape[0]/2), 
                                      replace=False)
            y_vision_test[oh_ind] = 'o'  # random replace half of 'zero' to 'oh'
        # Split data into disjoint support and query sets
        x_vision_test_split, y_vision_test_split = (
            data.make_train_test_split(x_vision_test_preprocess,
                                       y_vision_test_placeholder,
                                       test_ratio=0.5,
                                       shuffle=True,
                                       seed=ARGS.random_seed))
        
        # -----------------------------------------
        # Create mulitmodal few-shot test pipeline:
        # -----------------------------------------
        if ARGS.query_type == 'speech':
            speech_matching_set = False
            speech_queries = ARGS.n_queries
            vision_matching_set = True
            vision_queries = ARGS.l_way
        else:
            speech_matching_set = True
            speech_queries = ARGS.l_way
            vision_matching_set = False
            vision_queries = ARGS.n_queries
        # Create multimodal few-shot episode label set
        episode_label_set = data.create_episode_label_set(y_speech_test_split[0],
                                                          y_speech_test_split[1],
                                                          y_vision_test_split[0],
                                                          y_vision_test_split[1],
                                                          l_way=ARGS.l_way,
                                                          seed=ARGS.random_seed)
        # Batch episodes of support/query/matching sets for few-shot speech test
        speech_test_pipeline = (
            data.batch_few_shot_episodes(x_support_data=x_speech_test_split[0],
                                         y_support_labels=y_speech_test_split[0],
                                         z_support_originators=z_speech_test_split[0],
                                         x_query_data=x_speech_test_split[1],
                                         y_query_labels=y_speech_test_split[1],
                                         z_query_originators=z_speech_test_split[1],
                                         episode_label_set=episode_label_set,
                                         make_matching_set=speech_matching_set,
                                         k_shot=ARGS.k_shot,
                                         l_way=ARGS.l_way,
                                         n_queries=speech_queries,
                                         seed=ARGS.random_seed))
        # Batch episodes of support/query/matching sets for few-shot vision test
        vision_test_pipeline = (
            data.batch_few_shot_episodes(x_support_data=x_vision_test_split[0],
                                         y_support_labels=y_vision_test_split[0],
                                         x_query_data=x_vision_test_split[1],
                                         y_query_labels=y_vision_test_split[1],
                                         episode_label_set=episode_label_set,
                                         make_matching_set=vision_matching_set,
                                         k_shot=ARGS.k_shot,
                                         l_way=ARGS.l_way,
                                         n_queries=vision_queries,
                                         seed=ARGS.random_seed))
        speech_test_pipeline = speech_test_pipeline.prefetch(1)  # prefetch 1 batch per step
        vision_test_pipeline = vision_test_pipeline.prefetch(1)  # prefetch 1 batch per step
    
        # Create pipeline iterators
        speech_test_iterator = speech_test_pipeline.make_initializable_iterator()
        vision_test_iterator = vision_test_pipeline.make_initializable_iterator()
        test_feed_dict = {
            x_speech_test_placeholder: x_speech_test,
            y_speech_test_placeholder: y_speech_test,
            z_speech_test_placeholder: z_speech_test,
            x_vision_test_placeholder: x_vision_test,
            y_vision_test_placeholder: y_vision_test
        }
    
    # --------------------------------------------------------------------------
    # Build, train, and validate model:
    # --------------------------------------------------------------------------
    # Build speech model version from loaded model params dict
    speech_graph = tf.Graph()
    with speech_graph.as_default():  #pylint: disable=E1129
        speech_model_embedding, speech_embed_input, speech_train_flag, _, _ = (
            speech.build_speech_model(speech_model_params, training=False))
    # Build selected model version from loaded model params dict
    vision_graph = tf.Graph()
    with vision_graph.as_default():  #pylint: disable=E1129
        vision_model_embedding, vision_embed_input, vision_train_flag, _, _ = (
            vision.build_vision_model(vision_model_params, training=False))
    # Build few-shot 1-Nearest Neighbour memory comparison model
    query_input = tf.placeholder(TF_FLOAT, shape=[None, None])
    support_memory_input = tf.placeholder(TF_FLOAT, shape=[None, None])
    model_nn_memory = nearest_neighbour.fast_knn_cos(q_batch=query_input,
                                                     m_keys=support_memory_input,
                                                     k_nn=1,
                                                     normalize=True)
    # Check if test using Dynamic Time Warping instead of 1-NN for speech model
    test_dtw = False
    dtw_cost_func = None
    dtw_post_process = None
    if speech_model_params['model_version'] == 'dtw':
        test_dtw = True
        dtw_cost_func = speech_dtw.multivariate_dtw_cost_cosine
        dtw_post_process = lambda x: np.ascontiguousarray(  # as cython C-order
            np.swapaxes(  # time on x-axis for DTW
                _get_unpadded_image(np.squeeze(x, axis=-1), n_padding), 1, 0),
            dtype=float)
    # Check if test using pure pixel matching
    test_pixels = False
    if vision_model_params['model_version'] == 'pixels':
        test_pixels = True
    # Test the multimodal few-shot model
    test_mulitmodal_few_shot_model(
        test_feed_dict=test_feed_dict,
        # Speech test params ...
        speech_graph=speech_graph,
        speech_train_flag=speech_train_flag,
        speech_test_iterator=speech_test_iterator,
        speech_model_embedding=speech_model_embedding,                
        speech_embed_input=speech_embed_input,
        # Vision test params ...
        vision_graph=vision_graph,
        vision_train_flag=vision_train_flag,
        vision_test_iterator=vision_test_iterator,
        vision_model_embedding=vision_model_embedding,
        vision_embed_input=vision_embed_input,
        # Nearest neigbour params ...
        query_input=query_input,
        support_memory_input=support_memory_input,
        nearest_neighbour=model_nn_memory,
        n_episodes=ARGS.n_test_episodes,
        query_type=ARGS.query_type,
        test_pixels=test_pixels,
        test_dtw=test_dtw,
        dtw_cost_func=dtw_cost_func,
        dtw_post_process=dtw_post_process,
        # Other params ...
        log_interval=int(ARGS.n_test_episodes/10),
        model_dir=test_model_dir,
        speech_model_dir=speech_model_dir,
        vision_model_dir=vision_model_dir,
        summary_dir='summaries/test',
        speech_restore_checkpoint=ARGS.speech_restore_checkpoint,
        vision_restore_checkpoint=ARGS.vision_restore_checkpoint)


def test_mulitmodal_few_shot_model(
        test_feed_dict,
        # Speech test params ...
        speech_graph,
        speech_train_flag,
        speech_test_iterator,
        speech_model_embedding,                
        speech_embed_input,
        # Vision test params ...
        vision_graph,
        vision_train_flag,
        vision_test_iterator,
        vision_model_embedding,
        vision_embed_input,
        # Nearest neigbour params ...
        query_input,
        support_memory_input,
        nearest_neighbour,
        n_episodes,
        query_type,
        test_pixels=False,
        test_dtw=False,
        dtw_cost_func=None,
        dtw_post_process=None,
        # Other params ...
        log_interval=1,
        model_dir='saved_models',
        speech_model_dir='saved_models/speech',
        vision_model_dir='saved_models/vision',
        summary_dir='summaries/test',
        speech_restore_checkpoint=None,
        vision_restore_checkpoint=None):
    # Create tf.Session's for speech, vision, and general models
    speech_session = tf.Session(graph=speech_graph)
    vision_session = tf.Session(graph=vision_graph)
    general_session = tf.Session()  # default graph
    # Get model global steps
    with speech_graph.as_default():
        speech_global_step = tf.train.get_or_create_global_step()
        speech_step = 0
    with vision_graph.as_default():
        vision_global_step = tf.train.get_or_create_global_step()
        vision_step = 0
    # --------------------------------------------------------------------------
    # Load models (unless using DTW or pixel matching) and log some debug info:
    # --------------------------------------------------------------------------
    if not test_dtw:
        with speech_session.as_default(), speech_graph.as_default():  #pylint: disable=E1129
            try:  # restore speech from model checkpoint
                speech_checkpoint_saver = tf.train.Saver(
                    save_relative_paths=True)
                if speech_restore_checkpoint is not None:  # specific checkpoint
                    restore_path = os.path.join(speech_model_dir, 'checkpoints',
                        speech_restore_checkpoint)
                    if not os.path.isfile('{}.index'.format(restore_path)):
                        restore_path = speech_restore_checkpoint  # full path
                else:  # use best model if available
                    final_model_dir = os.path.join(speech_model_dir,
                        'final_model')
                    restore_path = tf.train.latest_checkpoint(final_model_dir)
                    if restore_path is None:
                        logging.info("No best model checkpoint could be found "
                                     "in directory: {}".format(final_model_dir))
                        return  # exit ... 
                speech_checkpoint_saver.restore(speech_session, restore_path)
                logging.info("Speech model restored from checkpoint: {}".format(restore_path))
            except ValueError:  # no checkpoints, inform and exit ...
                logging.info("Vision model checkpoint could not found at restore "
                            "path: {}".format(restore_path))
                return  # exit ...
            # Evaluate global speech model was trained to
            speech_step = speech_session.run(speech_global_step)
            logging.info("Testing speech model from: Global Step: {}"
                         "".format(speech_step))
    else:
        logging.info("Testing speech model with dynamic time warping (DTW).")

    if not test_pixels:
        with vision_session.as_default(), vision_graph.as_default():  #pylint: disable=E1129
            try:  # restore vision from model checkpoint
                vision_checkpoint_saver = tf.train.Saver(
                    save_relative_paths=True)
                if vision_restore_checkpoint is not None:  # specific checkpoint
                    restore_path = os.path.join(vision_model_dir, 'checkpoints',
                        vision_restore_checkpoint)
                    if not os.path.isfile('{}.index'.format(restore_path)):
                        restore_path = vision_restore_checkpoint  # full path
                else:  # use best model if available
                    final_model_dir = os.path.join(vision_model_dir,
                        'final_model')
                    restore_path = tf.train.latest_checkpoint(final_model_dir)
                    if restore_path is None:
                        logging.info("No best model checkpoint could be found "
                                     "in directory: {}".format(final_model_dir))
                        return  # exit ... 
                vision_checkpoint_saver.restore(vision_session, restore_path)
                logging.info("Vision model restored from checkpoint: {}".format(restore_path))
            except ValueError:  # no checkpoints, inform and exit ...
                logging.info("Speech model checkpoint could not found at restore "
                            "path: {}".format(restore_path))
                return  # exit ...
            # Evaluate global vision model was trained to
            vision_step = vision_session.run(vision_global_step)
            logging.info("Testing vision model from: Global Step: {}"
                         .format(vision_step))
    else:
        logging.info("Testing vision model with pure pixel matching.")
            
    # Create general session summary writer and few-shot accuracy summary
    summary_writer = tf.summary.FileWriter(os.path.join(
        model_dir, summary_dir, 
        datetime.datetime.now().strftime("%Hh%Mm%Ss_%f")),
        general_session.graph)
    # Get tf.summary tensor to evaluate for few-shot accuracy
    test_acc_input = tf.placeholder(TF_FLOAT)
    test_summ = tf.summary.scalar('test_few_shot_accuracy', test_acc_input)
    # Get speech and vision support/query/matching few-shot sets, and save 
    # one episode to tensorboard and files for debugging
    speech_support_set, speech_query_set = speech_test_iterator.get_next()
    vision_support_set, vision_query_set = vision_test_iterator.get_next()
    general_session.run([speech_test_iterator.initializer, 
                         vision_test_iterator.initializer],
                        feed_dict=test_feed_dict)  # init test set iterator
    speech_s_summ = tf.summary.image('speech_support_set_images', 
        speech_support_set[0], 10)
    vision_s_summ = tf.summary.image('vision_support_set_images',
        vision_support_set[0], 10)
    if query_type == 'speech':  # speech query, image matching set
        speech_q_summ = tf.summary.image('speech_query_set_images',
            speech_query_set[0], 10)
        vision_q_summ = tf.summary.image('vision_matching_set_images',
            vision_query_set[0], 10)
    else:  # image query, speech matching set
        speech_q_summ = tf.summary.image('speech_matching_set_images',
            speech_query_set[0], 10)
        vision_q_summ = tf.summary.image('vision_query_set_images',
            vision_query_set[0], 10)
    (speech_s_batch, speech_q_batch, vision_s_batch, vision_q_batch, 
     speech_s_images, speech_q_images, vision_s_images, vision_q_images) = (
        general_session.run([speech_support_set, speech_query_set,
                             vision_support_set, vision_query_set,
                             speech_s_summ, speech_q_summ,
                             vision_s_summ, vision_q_summ]))
    # Save figures to pdf for later use ...
    for index, (image, label, speaker) in enumerate(zip(*speech_s_batch)):
        if test_dtw:
            image = dtw_post_process(image)
        else:
            image = np.squeeze(image, axis=-1)
        utils.save_image(image, filename=os.path.join(model_dir,
            'multimodal_test_images', '{}_{}_{}_{}_{}_{}_{}.pdf'.format(
                'speech', 'support', index, 'label', label.decode("utf-8"),
                'speaker', speaker.decode("utf-8"))), cmap='inferno')
    for index, (image, label, speaker) in enumerate(zip(*speech_q_batch)):
        if test_dtw:
            image = dtw_post_process(image)
        else:
            image = np.squeeze(image, axis=-1)
        set_label = 'query' if query_type == 'speech' else 'matching'
        utils.save_image(image, filename=os.path.join(model_dir,
            'multimodal_test_images', '{}_{}_{}_{}_{}_{}_{}.pdf'.format(
                'speech', set_label, index, 'label', label.decode("utf-8"),
                'speaker', speaker.decode("utf-8"))), cmap='inferno')
    for index, (image, label) in enumerate(zip(*vision_s_batch)):
        utils.save_image(np.squeeze(image, axis=-1), filename=os.path.join(
            model_dir, 'multimodal_test_images', '{}_{}_{}_{}_{}.pdf'.format(
                'vision', 'support', index, 'label', label.decode("utf-8"))),
                cmap='gray_r')
    for index, (image, label) in enumerate(zip(*vision_q_batch)):
        set_label = 'query' if query_type == 'vision' else 'matching'
        utils.save_image(np.squeeze(image, axis=-1), filename=os.path.join(
            model_dir, 'multimodal_test_images', '{}_{}_{}_{}_{}.pdf'.format(
                'vision', set_label, index, 'label', label.decode("utf-8"))),
                cmap='gray_r')
    # Save summary images with general summary writer
    summary_writer.add_summary(speech_s_images, speech_step)
    summary_writer.add_summary(speech_q_images, speech_step)
    summary_writer.add_summary(vision_s_images, vision_step)
    summary_writer.add_summary(vision_q_images, vision_step)
    summary_writer.flush()
    
    # --------------------------------------------------------------------------
    # Cross-modal few-shot testing:
    # --------------------------------------------------------------------------
    total_queries = 0
    total_correct = 0
    general_session.run([speech_test_iterator.initializer,
                         vision_test_iterator.initializer], 
                        feed_dict=test_feed_dict)  # init test set iterator
    few_shot_set = [(speech_support_set, speech_query_set),
                    (vision_support_set, vision_query_set)]
    for episode in range(n_episodes):
        # Get next few-shot batch
        episode_batch = general_session.run(few_shot_set)
        speech_batch = episode_batch[0]
        vision_batch = episode_batch[1]
        # Get embeddings and classify queries with 1-NN on support set
        with speech_session.as_default(), speech_graph.as_default():  #pylint: disable=E1129
            speech_support_embeddings = speech_session.run(
                speech_model_embedding, 
                feed_dict={speech_embed_input: speech_batch[0][0],
                           speech_train_flag: False})
            speech_query_embeddings = speech_session.run(
                speech_model_embedding, 
                feed_dict={speech_embed_input: speech_batch[1][0],
                           speech_train_flag: False})
        with vision_session.as_default(), vision_graph.as_default():  #pylint: disable=E1129
            vision_support_embeddings = vision_session.run(
                vision_model_embedding, 
                feed_dict={vision_embed_input: vision_batch[0][0],
                           vision_train_flag: False})
            vision_query_embeddings = vision_session.run(
                vision_model_embedding, 
                feed_dict={vision_embed_input: vision_batch[1][0],
                           vision_train_flag: False})
        # Speech query cross-modal matching to images
        if query_type == 'speech':
            pred_message = ""
            if not test_dtw:  # test speech with fast cosine 1-NN memory model
                s_nearest_neighbour_indices = general_session.run(
                    nearest_neighbour,
                    feed_dict={query_input: speech_query_embeddings,
                               support_memory_input: speech_support_embeddings})
            else:  # test speech with dynamic time warping baseline
                costs = [[dtw_cost_func(
                    dtw_post_process(speech_query_embeddings[i]),
                    dtw_post_process(speech_support_embeddings[j]),
                    True) for j in range(len(speech_support_embeddings))]
                         for i in range(len(speech_query_embeddings))]
                s_nearest_neighbour_indices = [
                    np.argmin(costs[i]) for i in range(len(costs))]
            # Get cross-modal matches in image support set and their labels
            query_cross_matches = vision_support_embeddings[s_nearest_neighbour_indices]
            actual_labels = speech_batch[1][1]
            # Find cross-modal matches in the image matching set
            v_nearest_neighbour_indices = general_session.run(nearest_neighbour,
                feed_dict={query_input: query_cross_matches,
                           support_memory_input: vision_query_embeddings})
            predicted_labels = vision_batch[1][1][v_nearest_neighbour_indices]    
            # Images 'z' and 'o' treated same regardless of different speech classes
            actual_labels_update = np.array([  
                label if label != b'o' else b'z' for label in actual_labels])
            predicted_labels_update = np.array([
                label if label != b'o' else b'z' for label in predicted_labels])
            # Put together some debug info ...
            pred_message += "\t\tActual speech query labels:\t\t{}".format(
                actual_labels)
            pred_message += "\n\t\tPredicted speech support labels:\t{}".format(
                speech_batch[0][1][s_nearest_neighbour_indices])
            pred_message += "\n\t\tAssociated vision support labels:\t{}".format(
                vision_batch[0][1][s_nearest_neighbour_indices])
            pred_message += "\n\t\tPredicted vision matching labels:\t{}".format(
                predicted_labels)
            pred_message += "\n\t\tUpdated speech query labels ('o'=='z'):\t{}".format(
                actual_labels_update)
            pred_message += "\n\t\tUpdated vision match labels ('o'=='z'):\t{}".format(
                predicted_labels_update)
            # Update accuracy counters and log info
            total_correct += np.sum(actual_labels == predicted_labels)
            total_queries += speech_batch[1][1].shape[0]
            if episode % log_interval == 0:
                avg_acc = total_correct/total_queries
                ep_message = ("\tFew-shot Test: [Episode: {}/{}]\t"
                                "Average accuracy: {:.7f}".format(
                                    episode, n_episodes, avg_acc))
                logging.info(ep_message)
                logging.info(pred_message)
        # Image query cross-modal matching to speech
        else:
            pred_message = ""
            v_nearest_neighbour_indices = general_session.run(nearest_neighbour,
                feed_dict={query_input: vision_query_embeddings,
                            support_memory_input: vision_support_embeddings})
            # Get cross-modal matches in speech support set and their labels
            query_cross_matches = speech_support_embeddings[v_nearest_neighbour_indices]
            actual_labels = vision_batch[1][1]
            # Find cross-modal matches in the speech matching set
            if not test_dtw:  # test speech with fast cosine 1-NN memory model
                s_nearest_neighbour_indices = general_session.run(
                    nearest_neighbour,
                    feed_dict={query_input: query_cross_matches,
                               support_memory_input: speech_query_embeddings})
            else:  # test speech with dynamic time warping baseline
                costs = [[dtw_cost_func(
                    dtw_post_process(query_cross_matches[i]),
                    dtw_post_process(speech_query_embeddings[j]),
                    True) for j in range(len(speech_query_embeddings))]
                         for i in range(len(query_cross_matches))]
                s_nearest_neighbour_indices = [
                    np.argmin(costs[i]) for i in range(len(costs))]
            predicted_labels = speech_batch[1][1][s_nearest_neighbour_indices]
            # Images 'z' and 'o' treated same regardless of different speech classes
            actual_labels_update = np.array([
                label if label != b'o' else b'z' for label in actual_labels])
            predicted_labels_update = np.array([
                label if label != b'o' else b'z' for label in predicted_labels])
            # Put together some debug info ...
            pred_message += "\t\tActual vision query labels:\t\t{}".format(
                actual_labels)
            pred_message += "\n\t\tPredicted vision support labels:\t{}".format(
                vision_batch[0][1][v_nearest_neighbour_indices])
            pred_message += "\n\t\tAssociated speech support labels:\t{}".format(
                speech_batch[0][1][v_nearest_neighbour_indices])
            pred_message += '\n\t\tPredicted speech matching labels:\t{}'.format(
                predicted_labels)
            pred_message += "\n\t\tUpdated vision query labels:\t\t{}".format(
                actual_labels_update)
            pred_message += '\n\t\tUpdated speech prediction labels:\t{}'.format(
                predicted_labels_update)
            # Update accuracy counters and log info
            total_correct += np.sum(vision_batch[1][1] == predicted_labels)
            total_queries += vision_batch[1][1].shape[0]
            if episode % log_interval == 0:
                avg_acc = total_correct/total_queries
                ep_message = ("\tFew-shot Test: [Episode: {}/{}]\t"
                                "Average accuracy: {:.7f}".format(
                                    episode, n_episodes, avg_acc))
                logging.info(ep_message)
                logging.info(pred_message)

    # ------------------------------------------------------------------
    # Print stats:
    # ------------------------------------------------------------------
    avg_acc = total_correct/total_queries
    few_shot_message = ("Test set (few-shot): Average accuracy: "
                        "{:.5f}".format(avg_acc))
    logging.info(few_shot_message)
    test_summ = general_session.run(test_summ, 
        feed_dict={test_acc_input: avg_acc})
    summary_writer.add_summary(test_summ, 0)
    summary_writer.flush()
    with open(os.path.join(model_dir, 'test_result.txt'), 'w') as res_file:
        res_file.write("Test accuracy: {:.5f}".format(avg_acc))
    # Testing complete
    logging.info("Testing complete.")
    
    speech_session.close()
    vision_session.close()
    general_session.close()


def load_model_params(model_dir, params_file=None, modality='speech'):
    """Load JSON model params from a specified file or previous run."""
    # Find model parameters file location
    model_params_store_fn = os.path.join(model_dir, MODEL_PARAMS_STORE_FN)
    if params_file is not None:
        params_file = os.path.join(model_dir, params_file)
        if not os.path.exists(params_file):
            logging.info("Could not find specified model parameters file: "
                         "{}.".format(params_file))
            return None  # exit ...
        else:
            logging.info("Using stored model parameters file: "
                         "{}".format(params_file))
    elif os.path.exists(model_params_store_fn):
        params_file = model_params_store_fn
        logging.info("Using stored model parameters file: "
                     "{}".format(params_file))
    else:
        logging.info("Model parameters file {} could not be found!"
                     "".format(model_params_store_fn))
        return None  # exit ...
    # Load JSON parameters into model params dict 
    try:
        with open(params_file, 'r') as fp:
            model_params = json.load(fp)
        logging.info("Successfully loaded JSON model parameters!")
        logging.info("Testing {} model: version={}".format(
            modality, model_params['model_version']))
    except json.JSONDecodeError as ex:
        logging.info("Could not read JSON model parameters! "
                        "Caught exception: {}".format(ex))
        return None  # exit ...
    return model_params


def _get_unpadded_image(image, pad_length=110):
    """Remove padding from an end-padded mfcc/fbank image."""
    for i in range(pad_length-1, -1, -1):
        pad_length = i
        if np.sum(image[:, pad_length]) != 0.:
            break
    return image[:, :(pad_length+1)]


if __name__ == '__main__':
    # Call the script main function
    main()
    print('Exitting ...')