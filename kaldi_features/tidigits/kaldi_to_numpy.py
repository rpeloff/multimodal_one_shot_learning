#!/usr/bin/env python

"""
Convert a Kaldi archive to a NumPy archive.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018
"""

from __future__ import division
from __future__ import print_function
from datetime import datetime
from os import path
import argparse
import numpy as np
import struct
import sys

sys.path.append(path.join("..", "src", "utils"))

# import plotting


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("scp_fn", type=str, help="Kaldi SCP to read")
    parser.add_argument("npz_fn", type=str, help="NumPy archive to write to")    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


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

            tmp_mat = np.frombuffer(
                ark_read_buffer.read(rows*cols*4), dtype=np.float32
                )
            utt_mat = np.reshape(tmp_mat, (rows, cols))

            ark_read_buffer.close()

            ark_dict[utt_id] = utt_mat

    return ark_dict



#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    print(datetime.now())
    print("Reading:", args.scp_fn)
    features_dict = read_kaldi_ark_from_scp(args.scp_fn, ark_base_dir=".")
    print(datetime.now())
    print("No. utterances:", len(features_dict))

    print(datetime.now())
    print("Writing:", args.npz_fn)
    np.savez(args.npz_fn, **features_dict)
    print(datetime.now())

    # Plot example
    # import PIL.Image as Image
    # sys.path.append(path.join("..", "src", "utils"))
    # import plotting
    # plot_fn = path.splitext(args.npz_fn)[0] + "_eg.png"
    # print("Writing:", plot_fn)
    # first_utt = features_dict.keys()[0]
    # image = Image.fromarray(plotting.array_to_pixels(features_dict[first_utt]))
    # image.save(plot_fn)


if __name__ == "__main__":
    main()
