#!/usr/bin/env python

"""
Split a given subset into isolated (content) words.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017, 2018

Edited: Ryan Eloff
Date: October 2018
"""

from __future__ import print_function
from datetime import datetime
from os import path
from nltk.corpus import stopwords
import argparse
import numpy as np
import sys


min_dur = 0.40  # minimum duration in seconds
min_char = 0    # minimum number of characters in word token


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
        help="Directory containing flicker audio alignments (.ctm) file."
        )
    parser.add_argument(
        "datadir", type=str,
        help="Diretory containing feature data."
        )
    parser.add_argument(
        "features", type=str, help="feature set",
        choices=["fbank", "mfcc"]
        )
    parser.add_argument(
        "subset", type=str, help="subset to apply model to",
        choices=["dev", "test", "train"]
        )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def ctm_to_dict(ctm_fn):
    """
    Return a dictionary with a list of (start, dur, word) for each utterance.
    """
    ctm_dict = {}
    with open(ctm_fn, "r") as f:
        for line in f:
            utt, _, start, dur, word = line.strip().split(" ")
            if not utt in ctm_dict:
                ctm_dict[utt] = []
            start = float(start)
            dur = float(dur)
            if not "<" in word:
                ctm_dict[utt].append((start, dur, word.lower()))
    return ctm_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print(datetime.now())

    flickr8k_audio_dir = args.fadir
    data_dir = args.datadir

    ctm_fn = path.join(flickr8k_audio_dir, "flickr_8k.ctm")
    

    # # Keywords
    # print("Reading:", keywords_fn)
    # keywords = []
    # with open(keywords_fn) as f:
    #     for line in f:
    #         keywords.append(line.strip())
    # keywords = sorted(keywords)
    # print("No. keywords:", len(keywords))

    # Subset features
    features_dict_fn = path.join(args.datadir, args.features, args.subset + ".npz")
    print("Reading:", features_dict_fn)
    features_dict = np.load(features_dict_fn)
    utt_keys = sorted(features_dict.keys())
    print("No. {} utterances: {}".format(args.subset, len(utt_keys)))

    # # Get ~1000 utterances for queries, and the rest (~4000) for search
    # query_utt_keys = sorted(
    #     [utt_key for utt_key in utt_keys if int(utt_key[:3]) < 8]
    #     )
    # search_utt_keys = [
    #     utt_key for utt_key in utt_keys if utt_key not in query_utt_keys
    #     ]
    # print("No. query utterances:", len(query_utt_keys))
    # print("No. search utterances:", len(search_utt_keys))

    # Transcriptions
    print("Reading:", ctm_fn)
    ctm_dict = ctm_to_dict(ctm_fn)

    # Isolated words
    print(datetime.now())
    print("Getting isolated words")
    token_dict = {}
    for utt_key in sorted(utt_keys):
        ctm_key = utt_key[4:-2] + ".jpg_#" + utt_key[-1]
        ctm_entry = ctm_dict[ctm_key]
        utt_start = ctm_entry[0][0]
        utt_end = ctm_entry[-1][0] + ctm_entry[-1][1]
        for start, dur, word in ctm_entry:
            if (word not in stopwords.words("english") and dur >= min_dur and
                    len(word) >= min_char):
                i_start = int(np.round((start - utt_start)*100))
                i_end = i_start + int(np.round(dur * 100)) - 1
                token_key = "{}_{}_{:06d}-{:06d}".format(
                    word, utt_key, i_start, i_end
                    )
                token_dict[token_key] = features_dict[utt_key][
                    i_start:i_end, :
                    ]
    print("No. word tokens:", len(token_dict))
    print(datetime.now())

    tokens_dict_fn = path.join(
        args.datadir, args.features, args.subset + "_words.npz"
        )
    print("Writing:", tokens_dict_fn)
    np.savez(tokens_dict_fn, **token_dict)

    print(datetime.now())

    # # Queries
    # print(datetime.now())
    # print("Getting queries")
    # token_segments = []
    # query_dict = {}
    # keywords_covered = set()
    # for utt_key in sorted(query_utt_keys):
    # # for utt_key in sorted(utt_keys):  # for plotting embeddings
    #     ctm_key = utt_key[4:-2] + ".jpg_#" + utt_key[-1]
    #     ctm_entry = ctm_dict[ctm_key]
    #     utt_start = ctm_entry[0][0]
    #     utt_end = ctm_entry[-1][0] + ctm_entry[-1][1]
    #     for start, dur, word in ctm_entry:
    #         if word in keywords:
    #             i_start = int(np.round((start - utt_start)*100))
    #             i_end = i_start + int(np.round(dur * 100)) - 1
    #             query_key = "{}_{}_{:06d}-{:06d}".format(
    #                 word, utt_key, i_start, i_end
    #                 )
    #             query_dict[query_key] = features_dict[utt_key][
    #                 i_start:i_end, :
    #                 ]
    #             query_segments.append((query_key, utt_key, start, dur, word))
    #             keywords_covered.add(word)
    # print(
    #     "Keywords covered: {} out of {}".format(len(keywords_covered),
    #     len(keywords))
    #     )
    # print(datetime.now())
    # query_segments_fn = path.join(  
    #     "data", args.features, args.subset + "_query_segments.txt"
    #     )
    # print("Writing:", query_segments_fn)
    # with open(query_segments_fn, "w") as f:
    #     for query_key, utt_key, start, dur, word in query_segments:
    #         # print(query_segment)
    #         f.write("{} {} {} {} {}\n".format(
    #             query_key, word, utt_key, start, dur)
    #             )
    # query_dict_fn = path.join(
    #     "data", args.features, args.subset + "_queries.npz"
    #     )
    # print("Writing:", query_dict_fn)
    # np.savez(query_dict_fn, **query_dict)

    # # Plot some queries
    # # i = 0
    # # for query_key in query_dict:
    # #     if query_key.startswith("dog"):
    # #         import matplotlib.pyplot as plt
    # #         plt.figure()
    # #         plt.imshow(query_dict[query_key].T, interpolation="nearest")
    # #         i += 1
    # #     if i == 10:
    # #         break
    # # plt.show()

    # # Search collection
    # print(datetime.now())
    # print("Getting search collection")
    # search_dict = {}
    # search_segments = []
    # for utt_key in sorted(search_utt_keys):
    #     search_segments.append(utt_key)
    #     search_dict[utt_key] = features_dict[utt_key]
    # search_segments_fn = path.join(  
    #     "data", args.features, args.subset + "_search_segments.txt"
    #     )
    # print("Writing:", search_segments_fn)
    # with open(search_segments_fn, "w") as f:
    #     f.write("\n".join(search_segments))
    #     f.write("\n")
    # search_dict_fn = path.join(
    #     "data", args.features, args.subset + "_search.npz"
    #     )
    # print("Writing:", search_dict_fn)
    # np.savez(search_dict_fn, **search_dict)


    # print(datetime.now())


if __name__ == "__main__":
    main()
