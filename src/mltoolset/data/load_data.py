"""
Tools for loading common datasets.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: August 2018
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import mimetypes
import zipfile


from tqdm import tqdm
import numpy as np
import tensorflow as tf
import imageio
import requests


from .. import _globals


# URL to the Omniglot python raw archive files on GitHub 
OMNIGLOT_GITHUB_RAW_FILES = 'https://github.com/brendenlake/omniglot/raw/master/python/'


def load_mnist(path='mnist.npz'):
    """Load MNIST dataset wrapper (see tf.keras.datasets.mnist for more info)."""
    return tf.keras.datasets.mnist.load_data(path=path)


def load_omniglot(path='data/omniglot.npz'):
    """Load the Omniglot dataset from Brenden Lakes official repo.
    
    Data is returned as a tuple (background_set, evaluation_set), where each set
    consists of the tuple (x_data, y_labels, z_alphabet).

    Parameters
    ----------
    path : str
        Path to store cached dataset numpy archive.

    Returns
    -------
    omniglot : tuple of NumPy arrays
        Omniglot dataset returned as (background_set, evaluation_set).
    
    Notes
    -----
    Omniglot [1]_ is known as the inverse of MNIST as it contains many classes with
    few examples per class. 

    The images are 105x105 single channel arrays encoded with standard RGB color
    space. In other words, each pixel is an integer value in the range of 0-255,
    where 0 is black and 255 is white. The characters are represented by the 
    dark portions (0) of the image, whle the background is white (255).

    In contrast, MNIST is encoded with an inverse grayscale color map, where
    light portions (255) of the image represent the digit, and the background 
    is black (0). Thus if we want to pretrain a network on Omniglot in order to
    learn features relevant to MNIST, or vice versa, we would need to invert the
    values of MNIST.

    Invert operation for flattened 1D array shape (height*width):
    >>> x_data = list(map(lambda x: 255 - x, x_data))

    Or, for 2D image array of shape (height, width):
    >>> x_data = [list(map(lambda x: 255 - x, x_row)) for x_row in x_data]

    The datasets (background_set, evaluation_set) are broken down as follow:
    - background_set: 30 alphabets, 964 character classes, 19280 exemplars (20 per class).
    - evaluation_set: 20 alphabets, 659 character classes, 13180 exemplars (20 per class).
   
    Where each dataset consists of:
    - x_data: set of 105x105 Omniglot handwritten character images.
    - y_labels: set of image string labels (format: "{character_id}_{alphabet_index}").
    - z_alphabet: set of alphabets that characters are drawn from. 

    References
    ----------
    .. [1] B. M. Lake, R. Salakhutdinov, J. B. Tenenbaum (2015):
            Human-level concept learning through probabilistic program induction.
            http://www.sciencemag.org/content/350/6266/1332.short
            https://github.com/brendenlake/omniglot
    """ 
    omniglot = ()
    if os.path.isfile(path):
        np_data = np.load(path)
        omniglot += ((np_data['x_train'], np_data['y_train'], np_data['z_train']), )
        omniglot += ((np_data['x_test'], np_data['y_test'], np_data['z_test']), )
    else:
        print("Downloading Omniglot datasets ...")
        files = ['images_background.zip', 'images_evaluation.zip']
        for filename in files:
            # Download omniglot archives to temporary files
            file_url = OMNIGLOT_GITHUB_RAW_FILES + filename
            with open(filename, 'wb') as fp:
                response = requests.get(file_url, stream=True)
                total_length = response.headers.get('content-length')

                if total_length is None:  # No content length header
                    fp.write(response.content)
                else:
                    chunk_size = 1024  # 1 kB iterations
                    total_length = int(total_length)
                    n_chunks = int(np.ceil(total_length/chunk_size))
                    for _, data in zip(tqdm(range(n_chunks), unit='KB'), 
                                       response.iter_content(chunk_size=chunk_size)):
                        fp.write(data)  # Write data to temp file
            # Extract omniglot features from downloaded archives 
            x_data = []
            y_labels = []
            z_alphabets = []
            with zipfile.ZipFile(filename, 'r') as archive:
                arch_members = archive.namelist()
                for arch_member in arch_members:
                    if mimetypes.guess_type(arch_member)[0] == 'image/png':
                        # Split image path into parts to get label and alphabet
                        path_head, image_filename = os.path.split(arch_member)
                        path_head, character = os.path.split(path_head)
                        _, alphabet = os.path.split(path_head)
                        # Label is "{character_id}_{alphabet_index}"
                        label = "{}_{}".format(image_filename.split('_')[0],
                                               character.replace('character', ''))
                        # Extract and read image data array
                        with archive.open(arch_member) as extract_image:
                            image_data = imageio.imread(extract_image.read())
                        x_data.append(image_data)
                        y_labels.append(label)
                        z_alphabets.append(alphabet)
            os.remove(filename)  # Delete temporary archive files ... 
            omniglot += ((
                np.asarray(x_data, dtype=_globals.NP_INT),
                np.asarray(y_labels),
                np.asarray(z_alphabets)), )
        # Save downloaded and extracted data in numpy archive for later use
        dirname = os.path.dirname(path)
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)
        np.savez_compressed(path, 
                            x_train=omniglot[0][0],
                            y_train=omniglot[0][1],
                            z_train=omniglot[0][2],
                            x_test=omniglot[1][0],
                            y_test=omniglot[1][1],
                            z_test=omniglot[1][2])
    return omniglot


def load_flickraudio(
        path='flickr_audio.npz',
        feats_type='mfcc',
        encoding='latin1',
        remove_labels=None):
    """Load the Flickr-Audio extracted features."""
    valid_feats = ['mfcc', 'fbank']
    if feats_type not in valid_feats:
        raise ValueError("Invalid value specified for feats_type: {}. Expected "
                         "one of: {}.".format(feats_type, valid_feats))
    remove_labels = [] if remove_labels is None else remove_labels
    flickraudio = ()
    if os.path.isfile(path):
        np_data = np.load(path, encoding=encoding)[feats_type]  # select mfcc or fbanks
        # Add train words (and remove optional remove_labels words):
        train_labels = np.asarray(np_data[0][1], dtype=str)
        valid_ind = [i for i in range(len(train_labels))
                     if train_labels[i] not in remove_labels]
        flickraudio += ((np.ascontiguousarray(np_data[0][0])[valid_ind],  # features
                         train_labels[valid_ind],  # labels
                         np.asarray(np_data[0][2], dtype=str)[valid_ind],  # speakers
                         np.asarray(np_data[0][3], dtype=str)[valid_ind]), )  # segment_keys
        # Add dev words (and remove optional remove_labels words):
        dev_labels = np.asarray(np_data[1][1], dtype=str)
        valid_ind = [i for i in range(len(dev_labels))
                     if dev_labels[i] not in remove_labels]
        flickraudio += ((np.ascontiguousarray(np_data[1][0])[valid_ind],  # features
                         dev_labels[valid_ind],  # labels
                         np.asarray(np_data[1][2], dtype=str)[valid_ind],  # speakers
                         np.asarray(np_data[1][3], dtype=str)[valid_ind]), )  # segment_keys
        # Add test words (and remove optional remove_labels words):
        test_labels = np.asarray(np_data[2][1], dtype=str)
        valid_ind = [i for i in range(len(test_labels))
                     if test_labels[i] not in remove_labels]
        flickraudio += ((np.ascontiguousarray(np_data[2][0])[valid_ind],  # features
                         test_labels[valid_ind],  # labels
                         np.asarray(np_data[2][2], dtype=str)[valid_ind],  # speakers
                         np.asarray(np_data[2][3], dtype=str)[valid_ind]), )  # segment_keys
    else:
        raise ValueError("Cannot find Flickr-Audio data at specified path: {}."
                         "".format(path))
    return flickraudio


def load_tidigits(
        path='tidigits_audio.npz',
        feats_type='mfcc',
        encoding='latin1',
        dev_size=5000):
    """Load the TIDigits extracted features."""
    valid_feats = ['mfcc', 'fbank']
    if feats_type not in valid_feats:
        raise ValueError("Invalid value specified for feats_type: {}. Expected "
                         "one of: {}.".format(feats_type, valid_feats))
    tidigits = ()
    if os.path.isfile(path):
        np_data = np.load(path, encoding=encoding)[feats_type]  # select mfcc or fbanks
        # Add train words:
        tidigits += ((np_data[0][0][dev_size:],  # features
                      np.asarray(np_data[0][1][dev_size:], dtype=str),  # labels
                      np.asarray(np_data[0][2][dev_size:], dtype=str),  # speakers
                      np.asarray(np_data[0][3][dev_size:], dtype=str)), )  # segment_keys
        # Add dev words:
        tidigits += ((np_data[0][0][:dev_size],  # features
                      np.asarray(np_data[0][1][:dev_size], dtype=str),  # labels
                      np.asarray(np_data[0][2][:dev_size], dtype=str),  # speakers
                      np.asarray(np_data[0][3][:dev_size], dtype=str)), )  # segment_keys
        # Add train words:
        tidigits += ((np_data[1][0],  # features
                      np.asarray(np_data[1][1], dtype=str),  # labels
                      np.asarray(np_data[1][2], dtype=str),  # speakers
                      np.asarray(np_data[1][3], dtype=str)), )  # segment_keys
    else:
        raise ValueError("Cannot find TIDigits data at specified path: {}."
                         "".format(path))
    return tidigits


def preprocess_images(
        images,
        normalize=True,
        inverse_gray=False,
        resize_shape=(28, 28),
        resize_method=tf.image.ResizeMethod.BILINEAR,
        expand_dims=False,
        dtype=tf.float32):
    """Preprocess image data by resizing, normalizing, etc."""
    images_out = images
    if expand_dims:  # for single channel images without depth dimension
        images_out = tf.expand_dims(images_out, axis=-1)
    images_out = tf.image.resize_images(images_out, resize_shape, resize_method)
    images_out = tf.cast(images_out, dtype)
    if normalize:
        images_out = images_out / 255.
    if inverse_gray:  # requires that image is normalized!
        images_out = 1. - images_out
    return images_out


def pad_sequences(x, n_padded, center_padded=True):
    """Return the padded sequences and their original lengths."""
    padded_x = np.zeros((len(x), n_padded, x[0].shape[1]), dtype=_globals.NP_FLOAT)
    lengths = []
    for i_data, cur_x in enumerate(x):
        length = cur_x.shape[0]
        if center_padded:
            padding = int(np.round((n_padded - length) / 2.))
            if length <= n_padded:
                padded_x[i_data, padding:padding + length, :] = cur_x
            else:
                # Cut out snippet from sequence exceeding n_padded
                padded_x[i_data, :, :] = cur_x[-padding:-padding + n_padded]
            lengths.append(min(length, n_padded))
        else:
            length = min(length, n_padded)
            padded_x[i_data, :length, :] = cur_x[:length, :]
            lengths.append(length)
    return padded_x


# def shuffle_data(*data, seed=42):
#     shuffled_data = ()
#     for d in data:
#         shuffled_data += (tf.random_shuffle(d, seed=seed),)
#     return shuffled_data if len(data) > 1 else shuffled_data[0]
