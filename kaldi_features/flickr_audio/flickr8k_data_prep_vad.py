#!/usr/bin/env python

"""
Prepare the Flickr8k speech data, stripping out beginning and final silences.

Voice activity detection is based on the ground truth forced alignments
produced by David Harwath in CTM format.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017

Edited: Ryan Eloff
Date: October 2018
"""

from __future__ import absolute_import, division, print_function

from datetime import datetime
from os import path
import glob
import argparse
import numpy as np
import os
import re
import sys


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
        "outdir", type=str,
        help="Diretory to output prepared data."
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
            ctm_dict[utt].append((start, dur, word))
    return ctm_dict


def main():

    args = check_argv()

    flickr8k_audio_dir = args.fadir
    data_dir = args.outdir

    ctm_fn = path.join(flickr8k_audio_dir, "flickr_8k.ctm")
    wav_dir = path.join(flickr8k_audio_dir, "wavs")
    wav_to_spk_fn = path.join(flickr8k_audio_dir, "wav2spk.txt")

    print(datetime.now())

    # Directories
    if not path.isdir(data_dir):
        os.makedirs(data_dir)

    print("Reading:", ctm_fn)
    ctm_dict = ctm_to_dict(ctm_fn)

    # Speaker to utterance and vice versa
    spk_to_utt_dict = {}
    utt_to_spk_dict = {}
    speakers = set()
    utterances = []
    n_missing = 0
    with open(wav_to_spk_fn, "r") as f:
        for line in f:
            wav, speaker = line.strip().split(" ")
            speaker = "{:03d}".format(int(speaker))
            utt = speaker + "_" + path.splitext(wav)[0]
            ctm_label = utt[4:-2] + ".jpg_#" + utt[-1]
            if not ctm_label in ctm_dict:
                n_missing += 1
                continue
            if not speaker in spk_to_utt_dict:
                spk_to_utt_dict[speaker] = []
            spk_to_utt_dict[speaker].append(utt)
            utt_to_spk_dict[utt] = speaker
            speakers.add(speaker)
            utterances.append(utt)
    print("No. speakers:", len(speakers))
    spk_to_utt_fn = path.join(data_dir, "spk2utt")
    print("Writing:", spk_to_utt_fn)
    with open(spk_to_utt_fn, "w") as f:
        for speaker in sorted(spk_to_utt_dict):
            f.write(speaker + " " + " ".join(sorted(spk_to_utt_dict[speaker])) + "\n")
    utt_to_spk_fn = path.join(data_dir, "utt2spk")
    print("Writing:", utt_to_spk_fn)
    with open(utt_to_spk_fn, "w") as f:
        for utt in sorted(utt_to_spk_dict):
            f.write(utt + " " + utt_to_spk_dict[utt] + "\n")
    speakers_fn = path.join(data_dir, "speakers")
    print("Writing:", speakers_fn)
    with open(speakers_fn, "w") as f:
        for speaker in sorted(list(speakers)):
            f.write(speaker + "\n")

    wav_scp_fn = path.join(data_dir, "wav.scp")
    print("Writing:", wav_scp_fn)
    with open(wav_scp_fn, "w") as f:
        for utt in sorted(utterances):
            wav = path.join(wav_dir, utt[4:] + ".wav")
            f.write("{} sox {} -t wavpcm - |\n".format(utt, wav))

    # Segments and text
    segments_fn = path.join(data_dir, "segments")
    text_fn = path.join(data_dir, "text")
    print("Writing:", segments_fn)
    print("Writing:", text_fn)
    total_duration = 0
    with open(segments_fn, "w") as segments_f, open(text_fn, "w") as text_f:
        for utt in sorted(utterances):
            ctm_label = utt[4:-2] + ".jpg_#" + utt[-1]
            # if not ctm_label in ctm_dict:
            #     n_missing += 1
            #     continue
            ctm_entry = ctm_dict[utt[4:-2] + ".jpg_#" + utt[-1]]
            start = ctm_entry[0][0]
            end = ctm_entry[-1][0] + ctm_entry[-1][1]
            segments_f.write("{} {} {:.2f} {:.2f}\n".format(utt, utt, float(start), float(end)))
            text_f.write(utt + " " + " ".join([i[2].lower() for i in ctm_entry]) + "\n")
            total_duration += float(end) - float(start)
    print("No. alignments missing from CTM:", n_missing)
    print("Total duration: {:.2f} hours".format(total_duration/60.0/60.0))

    print(datetime.now())


if __name__ == "__main__":
    main()
