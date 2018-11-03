"""Training script for vision models.

Training, evaluation, and tuning scheme:

1. Use `python train_vision.py [options]` to build, train, and validate models.
2. Logs and model checkpoints will be stored in a unique dir for each specific
   run (e.g. .../{model_version}_{datetime}/), within the specified model dir.
3. To change base model parameters, either use the [options] interface, edit the
   param dicts in vision.py, or save the base params as json 
   (hint: `--save-base-params`), edit the file, and supply the model directory 
   containing the new params file to override the default base params.
4. To change training params and dataset choices, use the [options] interface of 
   the train script (see `--help` for more info).

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
matplotlib.use('Agg')  # use anitgrain rendering engine backend for non-GUI interfaces
from sklearn import preprocessing
import numpy as np
import tensorflow as tf


# Add upper-level 'src' directory to application path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


#pylint: disable=E0401
from mltoolset import training
from mltoolset import data
from mltoolset import nearest_neighbour
from mltoolset import utils
from mltoolset import TF_FLOAT, TF_INT
#pylint: enable=E0401
import vision


# Static names for stored log and param files
MODEL_PARAMS_STORE_FN = 'model_params.json'
LOG_FILENAME = 'train_vision.log'


# Training options passed by command line
# TODO(rpeloff) add to mltoolset training.py
TRAIN_OPTION_ARGS = [
    {   
        "varname": 'n_max_epochs',
        "names": ['-me', '--n-max-epochs'],
        "help": "Maximum number of epochs to train (default: 50)",
        "choices": None,
        "default": 50
    },
    {   
        "varname": 'batch_size',
        "names": ['-bs', '--batch-size'],
        "help": "Number of exemplars per mini-batch for SGD (default: 100)",
        "choices": None,
        "default": 100
    },
    {   
        "varname": 'optimizer',
        "names": ['-op', '--optimizer'],
        "help": "Optimizer for gradient descent (default: adam)",
        "choices": ['sgd', 'momentum', 'adagrad', 'adadelta', 'adam'],
        "default": 'adam'
    },
    {   
        "varname": 'learning_rate',
        "names": ['-lr', '--learning-rate'],
        "help": "Learning rate (default: 1e-3)",
        "choices": None,
        "default": 1e-3
    },
    {   
        "varname": 'decay_rate',
        "names": ['-dr', '--decay-rate'],
        "help": "Exponential decay of learning rate per epoch (default: 1.0 - no decay)",
        "choices": None,
        "default": 1.0
    },
    {   
        "varname": 'random_seed',
        "names": ['-rs', '--random-seed'],
        "help": "Random seed (default: 42)",
        "choices": None,
        "default": 42
    }
]


def check_arguments():
    """Check command line arguments for `python train_vision.py`."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])

    # --------------------------------------------------------------------------
    # General script options (model type, storage, etc.):
    # --------------------------------------------------------------------------
    parser.add_argument('--model-version',
                        type=str,
                        help="Type of vision model to build and train",
                        choices=vision.MODEL_VERSIONS,
                        required=True)
    parser.add_argument('--model-dir', 
                        type=os.path.abspath,
                        help="Path to store and load vision models "
                             "(defaults to the current working directory)",
                        default='.')
    parser.add_argument('--no-unique-dir',
                        action='store_true',
                        help="Do not create a unique model directory")
    parser.add_argument('--data-dir', 
                        type=os.path.abspath,
                        help="Path to store and load data"
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
    parser.add_argument('--restore-checkpoint', 
                        type=str,
                        help="Filename of model checkpoint to restore and "
                             "continue training (defaults to None which trains "
                             "from latest checkpoint if found)",
                        default=None)
    parser.add_argument('--save-base-params',
                        action='store_true',
                        help="Store the base parameters of the selected "
                             "vision model (as {}/{}) and exit".format(
                             '{model-dir}', MODEL_PARAMS_STORE_FN))
    
    # --------------------------------------------------------------------------
    # Model and data pipeline options:
    # --------------------------------------------------------------------------
    # Val size based on selecting first 10 out of 30 alphabets from Omniglot:
    # omni[0][2][np.isin(omni[0][2], np.unique(omni[0][2])[:10])].shape -> 6000
    parser.add_argument('--train-set',
                        type=str,
                        help="Dataset to use for pre-training vision model "
                             "(defaults to '{}')".format('omniglot'),
                        choices=['omniglot', 'mnist'],
                        default='omniglot')
    parser.add_argument('--val-size',
                        type=int,
                        help="Number of validation examples to select from "
                             "the train dataset (defaults to "
                             "{}; recommend {} for mnist)".format(6000, 5000),
                        default=6000)
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
    parser.add_argument('--n-train-episodes',
                        type=int,
                        help="Number of training episodes/batches per epoch "
                             "(defaults to {} which uses "
                             "train_size/batch_size)".format(None),
                        default=None),
    parser.add_argument('--n-test-episodes',
                        type=int,
                        help="Number of few-shot validation episodes"
                             "(defaults to {})".format(400),
                        default=400)
    parser.add_argument('--balanced-batching',
                        action='store_true',
                        help="Sample balanced batches of P concept classes and "
                             "K examples per class during training (defaults "
                             "to {}, which uses normal batching; Overrides "
                             "--batch-size with P_batch*K_batch)".format(False))
    parser.add_argument('--p-batch',
                        type=int,
                        help="Number of P_batch unique concept labels to "
                             "sample per balanced batch (defaults to {})"
                             "".format(32),
                        default=32)
    parser.add_argument('--k-batch',
                        type=int,
                        help="Number of K_batch examples to sample per unique "
                             "concept in a balanced batch (defaults to {})"
                             "".format(4),
                        default=4)
    parser.add_argument('--max-offline-pairs',
                       type=int,
                       help="Maximum number of same pairs that will be sampled "
                            "for siamese triplet (offline) model (defaults to "
                            "100000).",
                       default=int(100e3))

    # --------------------------------------------------------------------------
    # Other common training parameters:
    # --------------------------------------------------------------------------
    for train_opt in TRAIN_OPTION_ARGS:
        parser.add_argument(*train_opt['names'],
                            type=type(train_opt['default']),
                            choices=train_opt['choices'],
                            help=train_opt['help'],
                            default=train_opt['default'])
    
    # --------------------------------------------------------------------------
    # Vision model hyperparameters:
    # --------------------------------------------------------------------------
    for model_param in vision.base_model_dict.keys():
        parser.add_argument("--{}".format(model_param),
                            default=-1)  # default -1 to determine if value set

    return parser.parse_args()


def main():
    # --------------------------------------------------------------------------
    # Parse script args and handle options:
    # --------------------------------------------------------------------------
    ARGS = check_arguments()
    
    # Set numpy and tenorflow random seed
    np.random.seed(ARGS.random_seed)
    tf.set_random_seed(ARGS.random_seed)

    # Get specified model directory (default cwd)
    model_dir = ARGS.model_dir
    # Check if not using a previous run, and create a unique run directory
    if not os.path.exists(os.path.join(model_dir, LOG_FILENAME)):
        if not ARGS.no_unique_dir:
            unique_dir = "{}_{}_{}".format(
                'vision',
                ARGS.model_version, 
                datetime.datetime.now().strftime("%y%m%d_%Hh%Mm%Ss_%f"))
            model_dir = os.path.join(model_dir, unique_dir)

    # Create directories if required ...
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Set logging to print to console and log to file
    utils.set_logger(model_dir, log_fn=LOG_FILENAME)
    logging.info("Training vision model: version={}".format(ARGS.model_version))
    logging.info("Using model directory: {}".format(model_dir))

    # Save base parameters and exit if `--save-base-params` flag encountered
    if ARGS.save_base_params:
        base_params = vision.MODEL_BASE_PARAMS[ARGS.model_version].copy()
        base_params['model_version'] = ARGS.model_version
        base_params_path = os.path.join(model_dir, MODEL_PARAMS_STORE_FN)
        with open(base_params_path, 'w') as fp:
            logging.info("Writing base model parameters to file: {}"
                         "".format(base_params_path))
            json.dump(base_params, fp, indent=4)
        return  # exit ...

    # Load JSON model params from specified file or a previous run if available
    params_file = None
    model_params_store_fn = os.path.join(model_dir, MODEL_PARAMS_STORE_FN)
    if ARGS.params_file is not None:
        params_file = os.path.join(model_dir, ARGS.params_file)
        if not os.path.exists(params_file):
            logging.info("Could not find specified model parameters file: "
                         "{}.".format(params_file))
            return  # exit ...
        else:
            logging.info("Using stored model parameters file: "
                         "{}".format(params_file))
    elif os.path.exists(model_params_store_fn):
        params_file = model_params_store_fn
        logging.info("Using stored model parameters file: "
                     "{}".format(params_file))

    # If a model params file is found, load JSON into a model params dict 
    if params_file is not None:
        try:
            with open(params_file, 'r') as fp:
                model_params = json.load(fp)
            logging.info("Successfully loaded JSON model parameters!")
        except json.JSONDecodeError as ex:
            logging.info("Could not read JSON model parameters! "
                         "Caught exception: {}".format(ex))
            return  # exit ...
    else:
        # Get the default base model params for the specified model version
        model_params = vision.MODEL_BASE_PARAMS[ARGS.model_version].copy()
        logging.info("No model parameters file found. "
                     "Using base model parameters.")

    # Read and write training and model options from specified/default args
    train_options = {}
    var_args = vars(ARGS)
    for arg in var_args:
        if arg in vision.base_model_dict:
            if var_args[arg] != -1:  # if value explicitly set for model param
                model_params[arg] = var_args[arg]
        else:
            train_options[arg] = getattr(ARGS, arg)
    logging.info("Training parameters:")
    for train_opt, opt_val in train_options.items():
        logging.info("\t{}: {}".format(train_opt, opt_val))
    train_options_path = os.path.join(model_dir, 'train_options.json')
    with open(train_options_path, 'w') as fp:
        logging.info("Writing most recent training parameters to file: {}"
                        "".format(train_options_path))
        json.dump(train_options, fp, indent=4)
    
    # --------------------------------------------------------------------------
    # Add additional model parameters and save:
    # --------------------------------------------------------------------------
    image_size = 105 if (ARGS.train_set == 'omniglot') else 28
    model_params['model_version'] = ARGS.model_version  # for later rebuilding
    model_params_path = os.path.join(model_dir, MODEL_PARAMS_STORE_FN)
    with open(model_params_path, 'w') as fp:
        print("Writing model parameters to file: {}".format(model_params_path))
        json.dump(model_params, fp, indent=4)

    # For pixel matching model we simply want the model params, no training ...
    if ARGS.model_version == 'pixels':
        logging.info("Pure pixel matching model params ready for test!")
        return
    
    # --------------------------------------------------------------------------
    # Load pre-train dataset:
    # --------------------------------------------------------------------------
    if ARGS.train_set == 'omniglot':  # load omniglot (default) train set
        logging.info("Training vision model on dataset: {}".format('omniglot'))
        train_data = data.load_omniglot(
            path=os.path.join(ARGS.data_dir, 'omniglot.npz'))
        inverse_data = True  # inverse omniglot grayscale
    else:  # load mnist train set
        logging.info("Training vision model on dataset: {}".format('mnist'))
        train_data = data.load_mnist()
        inverse_data = False  # don't inverse mnist grayscale

    # --------------------------------------------------------------------------
    # Data processing pipeline (placed on CPU so GPU is free):
    # --------------------------------------------------------------------------
    with tf.device('/cpu:0'):
        # ------------------------------------
        # Create (pre-)train dataset pipeline:
        # ------------------------------------
        x_train = train_data[0][0][ARGS.val_size:].copy()
        y_train = train_data[0][1][ARGS.val_size:]
        x_train_placeholder = tf.placeholder(TF_FLOAT, 
                                             shape=[None, image_size, image_size])
        y_train_placeholder = tf.placeholder(TF_INT, shape=[None])
        # Get number of train batches/episodes per epoch
        if ARGS.n_train_episodes is not None:
            n_train_batches = ARGS.n_train_episodes
        else:
            n_train_batches = int(x_train.shape[0]/ARGS.batch_size)
        # Preprocess image data and labels
        x_train_preprocess = (
            data.preprocess_images(images=x_train_placeholder,
                                   normalize=True,
                                   inverse_gray=True,  # inverse omniglot
                                   resize_shape=model_params['resize_shape'],
                                   resize_method=tf.image.ResizeMethod.BILINEAR,
                                   expand_dims=True,
                                   dtype=TF_FLOAT))
        train_encoder = preprocessing.LabelEncoder()  # encode labels to indices
        y_train = train_encoder.fit_transform(y_train)
        model_params['n_output_logits'] = np.unique(y_train).shape[0]
        # y_train = tf.cast(y_train, TF_INT)
        # Shuffle data
        x_train_preprocess = tf.random_shuffle(x_train_preprocess,
                                               seed=ARGS.random_seed)
        y_train_preprocess = tf.random_shuffle(y_train_placeholder,
                                                seed=ARGS.random_seed)
        # Use balanced batching pipeline if specified, else batch full dataset
        if ARGS.balanced_batching:
            train_pipeline = (
                data.batch_k_examples_for_p_concepts(x_data=x_train_preprocess,
                                                     y_labels=y_train_preprocess,
                                                     p_batch=ARGS.p_batch,
                                                     k_batch=ARGS.k_batch,
                                                     seed=ARGS.random_seed))
#         elif ARGS.model_version == 'siamese_triplet':
#             train_triplets = data.sample_triplets(x_data=x_train_preprocess,
#                                                   y_labels=y_train_preprocess,
#                                                   use_dummy_data=True,
#                                                   n_max_same_pairs=ARGS.max_offline_pairs)
#             x_triplet_data = tf.data.Dataset.zip((
#                 tf.data.Dataset.from_tensor_slices(train_triplets[0][0]).batch(ARGS.batch_size, drop_remainder=True),
#                 tf.data.Dataset.from_tensor_slices(train_triplets[0][1]).batch(ARGS.batch_size, drop_remainder=True),
#                 tf.data.Dataset.from_tensor_slices(train_triplets[0][1]).batch(ARGS.batch_size, drop_remainder=True)))
#             y_triplet_data = tf.data.Dataset.zip((
#                 tf.data.Dataset.from_tensor_slices(train_triplets[1][0]).batch(ARGS.batch_size, drop_remainder=True),
#                 tf.data.Dataset.from_tensor_slices(train_triplets[1][1]).batch(ARGS.batch_size, drop_remainder=True),
#                 tf.data.Dataset.from_tensor_slices(train_triplets[1][1]).batch(ARGS.batch_size, drop_remainder=True)))
#             train_pipeline = tf.data.Dataset.zip((x_triplet_data, y_triplet_data))
#             n_train_batches = 1000  # not sure of batch size, loop until out of range ...
#             # Quick hack for num triplet batches in offline siamese ...
# #             with tf.Session() as sess:
# #                 n_triplets = sess.run(tf.shape(train_triplets[0])[1], feed_dict={
# #                     x_train_placeholder: x_train, y_train_placeholder: y_train})
# #                 n_train_batches = int(n_triplets/ARGS.batch_size)
# #                 logging.info("Calculated triplet batches: {} batches for batch size {} (total triplets: {})"
# #                              .format(n_train_batches, ARGS.batch_size, n_triplets))
        else:
            train_pipeline = data.batch_dataset(x_data=x_train_preprocess,
                                                y_labels=y_train_preprocess,
                                                batch_size=ARGS.batch_size,
                                                shuffle=True,
                                                seed=ARGS.random_seed,
                                                drop_remainder=True)
        # Triplet sampling from data pipeline for offline siamese models
        if (ARGS.model_version == 'siamese_triplet'):  
            train_pipeline = data.sample_dataset_triplets(train_pipeline,
                                                          use_dummy_data=True,
                                                          n_max_same_pairs=ARGS.max_offline_pairs)
        train_pipeline = train_pipeline.prefetch(1)  # prefetch 1 batch per step

        # --------------------------------------------
        # Create few-shot valdiation dataset pipeline:
        # --------------------------------------------
        x_val = train_data[0][0][ARGS.val_size:]
        y_val = train_data[0][1][ARGS.val_size:]
        x_val_placeholder = tf.placeholder(TF_FLOAT, 
                                             shape=[None, image_size, image_size])
        y_val_placeholder = tf.placeholder(tf.string, shape=[None])
        # Preprocess image data and labels
        x_val_preprocess = (
            data.preprocess_images(images=x_val_placeholder,
                                   normalize=True,
                                   inverse_gray=True,  # inverse omniglot
                                   resize_shape=model_params['resize_shape'],
                                   resize_method=tf.image.ResizeMethod.BILINEAR,
                                   expand_dims=True,
                                   dtype=TF_FLOAT))
        # y_val = tf.cast(y_val, TF_INT)
        # Split data into disjoint support and query sets
        x_val_split, y_val_split = (
            data.make_train_test_split(x_val_preprocess,
                                       y_val_placeholder,
                                       test_ratio=0.5,
                                       shuffle=True,
                                       seed=ARGS.random_seed))
        # Batch episodes of support and query sets for few-shot validation
        val_pipeline = (  #val_support_pipeline, val_query_pipeline = (
            data.batch_few_shot_episodes(x_support_data=x_val_split[0],
                                         y_support_labels=y_val_split[0],
                                         x_query_data=x_val_split[1],
                                         y_query_labels=y_val_split[1],
                                         k_shot=ARGS.k_shot,
                                         l_way=ARGS.l_way,
                                         n_queries=ARGS.n_queries,
                                         seed=ARGS.random_seed))
        val_pipeline = val_pipeline.prefetch(1)  # prefetch 1 batch per step

        # Create pipeline iterators and model train inputs
        train_iterator = train_pipeline.make_initializable_iterator()
        x_train_input, y_train_input = train_iterator.get_next()
        train_feed_dict = {
            x_train_placeholder: x_train,
            y_train_placeholder: y_train
        }
        val_iterator = val_pipeline.make_initializable_iterator()
        val_feed_dict = {
            x_val_placeholder: x_val,
            y_val_placeholder: y_val
        }

    # --------------------------------------------------------------------------
    # Build, train, and validate model:
    # --------------------------------------------------------------------------
    # Build selected model version from base/loaded model params dict
    model_embedding, embed_input, train_flag, train_loss, train_metrics = (
        vision.build_vision_model(model_params,
                                  x_train_data=x_train_input,
                                  y_train_labels=y_train_input))
    # Get optimizer and training operation specified in model params dict
    optimizer_class = utils.literal_to_optimizer_class(train_options['optimizer'])
    train_optimizer = training.get_training_op(
        optimizer_class=optimizer_class,
        loss_func=train_loss,
        learn_rate=train_options['learning_rate'],
        decay_rate=train_options['decay_rate'],
        n_epoch_batches=n_train_batches
    )
    # Build few-shot 1-Nearest Neighbour memory comparison model
    query_input = tf.placeholder(TF_FLOAT, shape=[None, None])
    support_memory_input = tf.placeholder(TF_FLOAT, shape=[None, None])
    model_nn_memory = nearest_neighbour.fast_knn_cos(q_batch=query_input,
                                                     m_keys=support_memory_input,
                                                     k_nn=1,
                                                     normalize=True)
    # Create tensorboard summaries
    tf.summary.scalar('loss', train_loss)
    for m_key, m_value in train_metrics.items():
        tf.summary.scalar(m_key, m_value)
    # Train the few-shot model
    train_few_shot_model(# Train params:
                         train_iterator=train_iterator,
                         train_feed_dict=train_feed_dict,
                         train_flag=train_flag,
                         train_loss=train_loss,
                         train_metrics=train_metrics,
                         train_optimizer=train_optimizer,
                         n_epochs=ARGS.n_max_epochs,
                         max_batches=n_train_batches,
                         # Validation params:
                         val_iterator=val_iterator,
                         val_feed_dict=val_feed_dict,
                         model_embedding=model_embedding,
                         embed_input=embed_input,
                         query_input=query_input,
                         support_memory_input=support_memory_input,
                         nearest_neighbour=model_nn_memory,
                         n_episodes=ARGS.n_test_episodes,
                         # Other params:
                         log_interval=int(n_train_batches/5),
                         model_dir=model_dir,
                         summary_dir='summaries/train',
                         save_filename='trained_model',
                         restore_checkpoint=ARGS.restore_checkpoint)


def train_few_shot_model(
        train_iterator,
        train_feed_dict,
        train_flag,
        train_loss,
        train_metrics,
        train_optimizer,
        n_epochs,
        max_batches,
        val_iterator,
        val_feed_dict,
        model_embedding,
        embed_input,
        query_input,
        support_memory_input,
        nearest_neighbour,
        n_episodes,
        # val_loss=None,  # None defaults to using train_loss for validation
        # val_metrics=None,  # None defaults to using train_metrics for validation
        # val_max_batches=None,  # None defaults to using max_batches for validation
        log_interval=1,  # Number of batches to complete between logging
        model_dir='saved_models',  # Saved models written to <model_dir>/checkpoints
        summary_dir='summaries/train',  # Directory for writing summaries
        save_filename='trained_model',  # Checkpoint filename
        restore_checkpoint=None  # Resumes training from a specific checkpoint
        ):
    # Get the global step
    global_step = tf.train.get_or_create_global_step()
    # Get tf.summary tensors to evaluate
    summaries = tf.summary.merge_all()
    val_acc_input = tf.placeholder(TF_FLOAT)
    val_summary = tf.summary.scalar('val_few_shot_accuracy', val_acc_input)
    # Define variables to store the best validation accuracy and epoch
    epoch_var = tf.Variable(0, name='best_epoch')
    accuracy_var = tf.Variable(0., name='best_accuracy')
    # Define a saver for model checkpoints
    checkpoint_saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)
    best_saver = tf.train.Saver(save_relative_paths=True)

    # Define helper function for saving model and parameters
    def _save_checkpoint(epoch, save_best=False, save_first=False):
        checkpoint_path = os.path.join(model_dir, 'checkpoints', save_filename)
        saved = checkpoint_saver.save(sess, checkpoint_path, global_step=epoch)
        logging.info("Saved model checkpoint to file (epoch {}): {}"
                     "".format(epoch, saved))
        if save_best:
            best_path = os.path.join(model_dir, 'final_model', save_filename)
            best_saved = best_saver.save(sess, best_path)
            logging.info("Saved new best model to file (epoch {}): {}"
                         "".format(epoch, best_saved))
            return best_saved
        elif save_first:
            first_path = os.path.join(model_dir, 'checkpoints', 'initial_model')
            first_saved = best_saver.save(sess, first_path)
            logging.info("Saved randomly initialized base model to file: {}"
                         "".format(first_saved))
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    # Start tf.Session to train and validate model
    with tf.Session() as sess:
        # ----------------------------------------------------------------------
        # Load model and log some debug info:
        # ----------------------------------------------------------------------
        try:  # restore from model checkpoint
            if restore_checkpoint is not None:  # use specific checkpoint
                restore_path = os.path.join(
                    model_dir, 'checkpoints', restore_checkpoint)
                if not os.path.isfile('{}.index'.format(restore_path)):
                    restore_path = restore_checkpoint  # possibly full path?
            else:  # use latest checkpoint if available
                restore_path = tf.train.latest_checkpoint(os.path.join(
                    model_dir, 'checkpoints'))
            checkpoint_saver.restore(sess, restore_path)
            logging.info("Model restored from checkpoint: {}".format(restore_path))
            start_epoch = int(restore_path.split('-')[-1])
        except ValueError:  # no checkpoints, initialize variables from scratch
            sess.run(tf.global_variables_initializer())
            start_epoch = 0
            _save_checkpoint(start_epoch, save_first=True)  # reproducibility
        
        # Evaluate starting global step, and previous best accuracy and epoch
        step = sess.run(global_step)
        logging.info("Training from: Epoch: {}\tGlobal Step: {}"
                     .format(start_epoch+1, step))
        best_epoch, best_val_acc = sess.run([epoch_var, accuracy_var])
        logging.info("Current best model: Epoch: {}\tValidation accuracy: "
                     "{:.5f}".format(best_epoch+1, best_val_acc))

        # Create session summary writer
        summary_writer = tf.summary.FileWriter(os.path.join(
            model_dir, summary_dir,
            datetime.datetime.now().strftime("%Hh%Mm%Ss_%f")), sess.graph)
        
        # TODO(rpeloff) If siamese ...
        # # Get some triplet pairs, and display on tensorboard
        # x_triplet, y_triplet = train_iterator.get_next()
        # sess.run(train_iterator.initializer)  # init validation set iterator
        # anch_summ = tf.summary.image('triplet_anchor_images', x_triplet[0], 5)
        # same_summ = tf.summary.image('triplet_same_images', x_triplet[1], 5)
        # diff_summ = tf.summary.image('triplet_different_images', x_triplet[2], 5)
        # triplet_batch, anch_images, same_images, diff_images = sess.run(
        #     [x_triplet, anch_summ, same_summ, diff_summ])
        # summary_writer.add_summary(anch_images, step)
        # summary_writer.add_summary(same_images, step)
        # summary_writer.add_summary(diff_images, step)
        # summary_writer.flush()

        # Get support/query few-shot set, and display one episode on tensorboard
        support_set, query_set = val_iterator.get_next()
        sess.run(val_iterator.initializer, feed_dict=val_feed_dict)  # init validation set iterator
        s_summ = tf.summary.image('support_set_images', support_set[0], 10)
        q_summ = tf.summary.image('query_set_images', query_set[0], 10)
        support_batch, query_batch, s_images, q_images = sess.run(
            [support_set, query_set, s_summ, q_summ])
        summary_writer.add_summary(s_images, step)
        summary_writer.add_summary(q_images, step)
        summary_writer.flush()
        # Save figures to pdf for later use ...
        for image, label in zip(*support_batch):
            utils.save_image(np.squeeze(image, axis=-1), filename=os.path.join(
                model_dir, 'train_images', '{}_{}_{}.pdf'.format('support', 'l',
                label.decode("utf-8"))), cmap='gray_r')
        for image, label in zip(*query_batch):
            utils.save_image(np.squeeze(image, axis=-1), filename=os.path.join(
                model_dir, 'train_images', '{}_{}_{}.pdf'.format('query', 'l', 
                label.decode("utf-8"))), cmap='gray_r')

        # ----------------------------------------------------------------------
        # Training:
        # ----------------------------------------------------------------------
        for epoch in range(start_epoch, n_epochs):
            logging.info("Epoch: {}/{} [Step: {}]"
                         "".format(epoch+1, n_epochs, step))
            sess.run(train_iterator.initializer, feed_dict=train_feed_dict)  # init train dataset iterator
            avg_loss = 0.
            n_batches_completed = 0
            
            if max_batches is None:
                max_iterator = zero_to_infinity_generator
            else:
                max_iterator = range(max_batches)
            for i in max_iterator:
                try:
                    # TODO(reploff) Add embeddings and labels to visualize how 
                    # embeddings change over training?
                    _, loss_val, summary_vals, metric_vals, step = sess.run(
                        [train_optimizer, train_loss, summaries,
                        [m for m in train_metrics.values()], global_step],
                        feed_dict={train_flag: True}, options=run_options)
                    
                    # Write summaries for tensorboard and log some info
                    summary_writer.add_summary(summary_vals, step)
                    summary_writer.flush()
                    n_batches_completed += 1
                    avg_loss += loss_val
                    if n_batches_completed % log_interval == 0:
                        batch_message = ("\tTrain: [Batch: {}/{}]\tLoss: {:.7f}"
                                         "".format(n_batches_completed, 
                                         max_batches, loss_val))
                        for metric_key, metric_val in zip(
                                [k for k in train_metrics.keys()], metric_vals):
                            batch_message += "\t{}: {}".format(metric_key,
                                                               metric_val)
                        logging.info(batch_message)      
                except tf.errors.OutOfRangeError:  # catch pipeline out of range
                    break
            # ------------------------------------------------------------------
            # Few-shot validation:
            # ------------------------------------------------------------------
            total_queries = 0
            total_correct = 0
            sess.run(val_iterator.initializer, feed_dict=val_feed_dict)  # init validation set iterator
            for episode in range(n_episodes):    
                support_batch, query_batch = sess.run([support_set, query_set])
                # Get embeddings and classify queries with 1-NN on support set
                support_embeddings = sess.run(model_embedding, 
                    feed_dict={embed_input: support_batch[0], train_flag: False})
                query_embeddings = sess.run(model_embedding, 
                    feed_dict={embed_input: query_batch[0], train_flag: False})
                nearest_neighbour_indices = sess.run(nearest_neighbour,
                    feed_dict={query_input: query_embeddings,
                               support_memory_input: support_embeddings})
                # Calculate and store number of correct predictions
                predicted_labels = support_batch[1][nearest_neighbour_indices] 
                total_correct += np.sum(query_batch[1] == predicted_labels)
                total_queries += query_batch[1].shape[0]
                if episode % int(n_episodes/5) == 0:
                    avg_acc = total_correct/total_queries
                    ep_message = ("\tFew-shot Test: [Episode: {}/{}]\t"
                                  "Average accuracy: {:.7f}".format(
                                      episode, n_episodes, avg_acc))
                    logging.info(ep_message)
            # ------------------------------------------------------------------
            # Print stats and early-stopping:
            # ------------------------------------------------------------------
            # Print epoch train stats
            avg_loss = avg_loss / n_batches_completed
            epoch_message = ("Epoch: {}/{} [Step: {}]\tTrain set: Average "
                             "loss: {:.5f}".format(epoch+1, n_epochs, step, 
                             avg_loss))
            logging.info(epoch_message)
            # Print epoch few-shot validation stats
            avg_acc = total_correct/total_queries
            few_shot_message = ("Epoch: {}/{} [Step: {}]\tValidation set (few-"
                                "shot): Average accuracy: {:.5f}".format(
                                epoch+1, n_epochs, step, avg_acc))
            logging.info(few_shot_message)
            val_summ = sess.run(val_summary, feed_dict={val_acc_input: avg_acc})
            summary_writer.add_summary(val_summ, step)
            summary_writer.flush()
            # Check if this is a new best model
            if avg_acc > best_val_acc:
                best_epoch = epoch
                best_val_acc = avg_acc
                sess.run([tf.assign(epoch_var, best_epoch),
                          tf.assign(accuracy_var, best_val_acc)])
                _save_checkpoint(epoch+1, save_best=True)  # save best model
                with open(os.path.join(model_dir, 'train_result.txt'), 'w') as res_file:
                    res_file.write("Epoch: {}\tTrain loss: {:.5f}\t" 
                                   "Validation accuracy: {:.5f}"
                                   .format(epoch+1, avg_loss, avg_acc))
        # Training complete, print final (best) model stats:
        logging.info("Training complete. Best model found at epoch {} with "
                     "validation accuracy {:.5f}.".format(best_epoch+1, best_val_acc))

        
def zero_to_infinity_generator():
    i = 0
    while True:
        yield i
        i += 1
        
if __name__ == '__main__':
    # Call the script main function
    main()
    print('Exitting ...')
