"""
Functions for dealing with data input and output.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from os import path
import numpy as np
import struct

NP_DTYPE = np.float32
NP_ITYPE = np.int32


def read_kaldi_ark_from_scp(scp_fn, ark_base_dir=""):
    """
    Read a binary Kaldi archive and return a dict of Numpy matrices, with the
    utterance IDs of the SCP as keys. Based on the code:
    https://github.com/yajiemiao/pdnn/blob/master/io_func/kaldi_feat.py

    Parameters
    ----------
    ark_base_dir : str
        The base directory for the archives to which the SCP points.
    """

    ark_dict = {}

    with open(scp_fn) as f:
        for line in f:
            if line == "":
                continue
            utt_id, path_pos = line.replace("\n", "").split(" ")
            ark_path, pos = path_pos.split(":")

            ark_path = path.join(ark_base_dir, ark_path)

            ark_read_buffer = open(ark_path, "rb")
            ark_read_buffer.seek(int(pos),0)
            header = struct.unpack("<xcccc", ark_read_buffer.read(5))
            assert header[0] == "B", "Input .ark file is not binary"

            rows = 0
            cols = 0
            m, rows = struct.unpack("<bi", ark_read_buffer.read(5))
            n, cols = struct.unpack("<bi", ark_read_buffer.read(5))

            tmp_mat = np.frombuffer(ark_read_buffer.read(rows*cols*4), dtype=np.float32)
            utt_mat = np.reshape(tmp_mat, (rows, cols))

            ark_read_buffer.close()

            ark_dict[utt_id] = utt_mat

    return ark_dict


def pad_sequences(x, n_padded, center_padded=True):
    """Return the padded sequences and their original lengths."""
    padded_x = np.zeros((len(x), n_padded, x[0].shape[1]), dtype=NP_DTYPE)
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

        # if length > n_padded:
        #     import PIL.Image as Image
        #     import sys
        #     sys.path.append(path.join("..", "tflego", "utils"))
        #     import plotting
        #     print cur_x.shape
        #     print padded_x[i_data, :5]
        #     print
        #     print cur_x[55:60]
        #     image = Image.fromarray(plotting.array_to_pixels(cur_x))
        #     image.save("1.png")
        #     image = Image.fromarray(plotting.array_to_pixels(padded_x[i_data]))
        #     image.save("2.png")
        #     assert False

    return padded_x, lengths


def load_flickr8k_padded_bow_labelled(data_dir, subset, n_padded, label_dict,
        word_to_id, center_padded=True):
    """
    Return the padded Flickr8k speech matrices and bag-of-word label vectors.

    Only words in `word_to_id` is labelled. The bag-of-word vectors contain 1
    for a word that occurs and 0 for a word that does not occur.
    """

    assert subset in ["train", "dev", "test"]

    # Load data and shuffle
    npz_fn = path.join(data_dir, subset + ".npz")
    print "Reading: " + npz_fn
    features_dict = np.load(npz_fn)
    utterances = sorted(features_dict.keys())
    np.random.shuffle(utterances)
    x = [features_dict[i] for i in utterances]

    # Get lengths and pad
    padded_x, lengths = pad_sequences(x, n_padded, center_padded)

    # Get bag-of-word vectors
    bow_vectors = np.zeros((len(x), len(word_to_id)), dtype=NP_DTYPE)
    for i_data, utt in enumerate(utterances):
        for word in label_dict[utt]:
            if word in word_to_id:
                bow_vectors[i_data, word_to_id[word]] = 1
        # print utt
        # print label_dict[utt]
        # print [word_to_id[word] for word in label_dict[utt] if word in word_to_id]
        # print dict([(i[1], i[0]) for i in word_to_id.iteritems()])[94]
        # print bow_vectors[i_data]
        # assert False

    return padded_x, bow_vectors, np.array(lengths, dtype=NP_DTYPE)


def load_flickr8k_padded_visionsig(speech_data_dir, subset, n_padded,
        visionsig_dict, d_visionsig, sigmoid_threshold=None,
        center_padded=True, tmp=None):
    """
    Return the padded Flickr8k speech matrices and vision sigmoid vectors.
    """

    assert subset in ["train", "dev", "test"]

    # Load data and shuffle
    npz_fn = path.join(speech_data_dir, subset + ".npz")
    print "Reading: " + npz_fn
    features_dict = np.load(npz_fn)
    utterances = sorted(features_dict.keys())
    np.random.shuffle(utterances)
    x = [features_dict[i] for i in utterances]

    # Get lengths and pad
    padded_x, lengths = pad_sequences(x, n_padded, center_padded)

    # Get vision sigmoids
    visionsig_vectors = np.zeros((len(x), d_visionsig), dtype=NP_DTYPE)
    for i_data, utt in enumerate(utterances):
        image_key = utt[4:-2]
        if sigmoid_threshold is None:
            visionsig_vectors[i_data, :] = visionsig_dict[image_key][:d_visionsig]
        else:
            visionsig_vectors[i_data, np.where(visionsig_dict[image_key][:d_visionsig] >= \
                sigmoid_threshold)[0]] = 1

    # # Get bag-of-word vectors
    # bow_vectors = np.zeros((len(x), len(word_to_id)), dtype=NP_DTYPE)
    # for i_data, utt in enumerate(utterances):
    #     for word in label_dict[utt]:
    #         if word in word_to_id:
    #             bow_vectors[i_data, word_to_id[word]] = 1
    #     # print utt
    #     # print label_dict[utt]
    #     # print [word_to_id[word] for word in label_dict[utt] if word in word_to_id]
    #     # print dict([(i[1], i[0]) for i in word_to_id.iteritems()])[94]
    #     # print bow_vectors[i_data]
    #     # assert False

    return padded_x, visionsig_vectors, np.array(lengths, dtype=NP_DTYPE)
