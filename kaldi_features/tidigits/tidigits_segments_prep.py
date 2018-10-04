#!/usr/bin/env python

"""
Use forced alignments to separate digit sequences into individual digits.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018

Edited: Ryan Eloff
Date: June 2018
"""

from __future__ import absolute_import, division, print_function
from os import path
import argparse
import sys

import numpy as np


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "fadir", type=str,
        help="Directory containing forced alignments."
        )
    parser.add_argument(
        "outdir", type=str,
        help="Diretory to write output individual segment files"
        )
    parser.add_argument(
        "dataset", type=str, choices={"train", "test"},
        help="Dataset to obtain segments for."
        )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    segments = {}

    # Read forced alignment
    fa_dir = args.fadir
    fa_fn = path.join(fa_dir, args.dataset + "_word_align.ctm")
    print("Reading:", fa_fn)
    
    with open(fa_fn) as f:
        # Keep track of the number of each digit key added per utterance sequence (in case of duplicates with the same key)
        utt_digit_keys = {}
        # For each entry in the forced alignments
        for line in f:

            # Create a segments entry
            line = line.split()
            utt_key = line[0]
            digit_start = float(line[2])
            digit_duration = float(line[3])
            digit_key = line[4]
            # Keep track of the number of each digit key per utterance key (in case of duplicate digit keys in same sequence)
            if not utt_key in utt_digit_keys:
                utt_digit_keys[utt_key] = {}
            if not digit_key in utt_digit_keys[utt_key]:
                utt_digit_keys[utt_key][digit_key] = 0
            else:
                utt_digit_keys[utt_key][digit_key] += 1
            # Change digit key to be '<digit_key>a' for first occurence, '<digit_key>b' for second occurence, etc.
            digit_key = digit_key + chr(utt_digit_keys[utt_key][digit_key] + ord('a'))
            
            segments[utt_key + "_" + digit_key] = (
                # NOTE: Previously used `extract-rows` Kaldi module to extract individual digit features with segment start/end specified in frames (time/[10 ms frame-shift] -> time * 100).
                # Kaldi now uses `extract-feature-segments` instead, with segment start/end specified in seconds. 
                # Integer floor of (time*100) maintained to generate same feature segments as previously.
		        utt_key, int(np.floor(digit_start*100))/100.0,
                int(np.floor((digit_start + digit_duration)*100))/100.0
                )

    # Write segments
    segments_fn = path.join(args.outdir, "segments_indiv")
    print("Writing:", segments_fn)
    with open(segments_fn, "w") as f:
        for segment_key in sorted(segments):
            utt_key, digit_start, digit_end = segments[segment_key]
            f.write(
                "{} {} {} {}\n".format(segment_key, utt_key, digit_start,
                digit_end)
                )


if __name__ == "__main__":
    main()