#!/usr/bin/env python

"""Combine individual MFCC and FBank TIDigits archives into single archive.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: August 2018
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import argparse


import numpy as np


def check_arguments():
    """Check command line arguments for `python combine_archives.py`."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    parser.add_argument('--mfcc-train',
                        type=str,
                        help="MFCC train set NumPy archive",
                        required=True)
    parser.add_argument('--mfcc-test',
                        type=str,
                        help="MFCC test set NumPy archive",
                        required=True)
    parser.add_argument('--fbank-train',
                        type=str,
                        help="Filterbank train set NumPy archive",
                        required=True)
    parser.add_argument('--fbank-test',
                        type=str,
                        help="Filterbank test set NumPy archive",
                        required=True)
    parser.add_argument('--out-file',
                        type=str,
                        help="Combined NumPy archive filename",
                        required=True)  

    return parser.parse_args()


def main():
    ARGS = check_arguments()
    
    feats_files = [(ARGS.mfcc_train, ARGS.mfcc_test), 
                   (ARGS.fbank_train, ARGS.fbank_test)]

    all_feats = ()
    for feats_type in feats_files:
        next_feats_data = ()
        for feats_file in feats_type:
            print("Reading archive: {}".format(feats_file))
            data = np.load(feats_file)
            k_keys = np.asarray(data.keys(), dtype=str)
            x_features = [data[key] for key in k_keys]  # stored as list due to varying length sequences
            y_labels = np.asarray(
                [key[-2] for key in k_keys], dtype=str)
            z_speakers = np.asarray(
                [key.split('_')[0] for key in k_keys], dtype=str)
            next_feats_data += ((x_features, y_labels, z_speakers, k_keys), )
            print("No. utterances:", len(x_features))
        all_feats += (next_feats_data, )

    print("Writing combined data to archive: {}".format(ARGS.out_file))
    np.savez(ARGS.out_file, mfcc=all_feats[0], fbank=all_feats[1])


if __name__ == "__main__":
    main()
