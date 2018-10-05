#!/bin/bash

# Script to extract speech features from within the reloff/kaldi:5.4 container.
#
# Author: Ryan Eloff
# Contact: ryan.peter.eloff@gmail.com
# Date: September 2018

# ------------------------------------------------------------------------------
# Install pip and required python packages
# ------------------------------------------------------------------------------
apt-get update -y \
    && apt-get install -y --no-install-recommends python-pip \
    && python -m pip install --upgrade setuptools wheel \
    && python -m pip install \
        numpy==1.15.0 \
        pillow==5.3.0 \
        matplotlib==2.2.3 \
        nltk==3.3.0
python -c "import nltk; nltk.download('stopwords')"

# ------------------------------------------------------------------------------
# Prepare Kaldi tools for feature extraction:
# ------------------------------------------------------------------------------
# Copy Kaldi example scripts to /tmp/kaldi
mkdir -p /tmp/kaldi
cp -rL /kaldi/egs/tidigits/s5/* /tmp/kaldi/  # from tidigits not wsj for scripts in local/
cd /tmp/kaldi  # work in temporary directory
# Set Kaldi train and decode commands (see cmd.sh kaldi/egs/wsj/s5)
echo '
    export train_cmd=run.pl
    export decode_cmd=run.pl
    export mkgraph_cmd=run.pl' > cmd.sh
# Set Kaldi tools path (from updated path.sh in kaldi/egs/wsj/s5)
echo '
    export KALDI_ROOT=/kaldi
    [ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
    export PATH=/tmp/kaldi/utils:$KALDI_ROOT/tools/openfst/bin:/tmp/kaldi:$PATH
    [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] \
        && echo >&2 "The standard file ${KALDI_ROOT}/tools/config/common_path.sh is not present -> Exit!" \
        && exit 1
    . $KALDI_ROOT/tools/config/common_path.sh
    export LC_ALL=C' > path.sh
source cmd.sh
source path.sh  # many scripts depend on this file being present in cwd

# ==============================================================================
#                                    TIDIGITS
# ------------------------------------------------------------------------------
# Prepare feature data directories:
# ------------------------------------------------------------------------------
tidigits_feats=$FEATURES_DIR/tidigits
mkdir -p $tidigits_feats/features
mkdir -p $tidigits_feats/logs

# ------------------------------------------------------------------------------
# Prepare TIDigits in the format required for Kaldi feature extraction:
# Based on tidigits_data_prep.sh in kaldi/egs/tidigits/s5
#   Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#   Apache 2.0.
# ------------------------------------------------------------------------------
# Create TIDIgits temporary data directories
tmpdir=/tmp/kaldi/tidigits
mkdir -p $tmpdir/train
mkdir -p $tmpdir/test
mkdir -p $tmpdir/local/data

# Make sure that tidigits data exists in container at /tidigits/tidigits
rootdir=/tidigits/tidigits  # set tidigits root data directory
if ! [ -d $rootdir ]; then 
    echo "Tidigits directory does not have expected format: ${rootdir}"
    exit 1  
fi

# Get train file list and check that we have the expected number (12549)
# Note: original script expects 8623 which considered only adult men and women
find $rootdir/train -name '*.wav' > $tmpdir/local/data/train.flist
n=`cat $tmpdir/local/data/train.flist | wc -l`
[ $n -eq 12549 ] || echo Unexpected number of training files $n versus 12549
# Get test file list and check that we have the expected number (12547) 
# Note: original script expects 8700 which considered only adult men and women
find $rootdir/test -name '*.wav' > $tmpdir/local/data/test.flist
n=`cat $tmpdir/local/data/test.flist | wc -l`
[ $n -eq 12547 ] || echo Unexpected number of test files $n versus 12547

# Get Kaldi sph2pipe program to convert data from "sphere" format to .wav format
sph2pipe=/kaldi/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe"
   exit 1
fi

# Prepare train and test data: convert to .wav format and extract meta-info
for set in train test; do
    # Get scp file that has utterance-ids and maps to the sphere file
    cat $tmpdir/local/data/$set.flist \
        | perl -ane 'm|/(..)/([1-9zo]+[ab])\.wav| || die "bad line $_"; print "$1_$2 $_"; ' \
        | sort > $tmpdir/local/data/${set}_sph.scp
    # Turn it into a valid .wav format (i.e. modern RIFF format, not sphere)
    awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' \
        < $tmpdir/local/data/${set}_sph.scp \
        > $tmpdir/$set/wav.scp
    # Get the "text" file that says what the transcription is
    cat $tmpdir/$set/wav.scp \
        | perl -ane 'm/^(.._([1-9zo]+)[ab]) / || die; $text = join(" ", split("", $2)); print "$1 $text\n";' \
        < $tmpdir/$set/wav.scp \
        > $tmpdir/$set/text
    # Get the "utt2spk" file that says, for each utterance, the speaker name
    perl -ane 'm/^((..)_\S+) / || die; print "$1 $2\n"; ' \
        < $tmpdir/$set/wav.scp \
        > $tmpdir/$set/utt2spk
    # Create file that maps from speaker to utterance-list
    /tmp/kaldi/utils/utt2spk_to_spk2utt.pl \
        < $tmpdir/$set/utt2spk \
        > $tmpdir/$set/spk2utt
done
echo "TIDigits data preparation succeeded!"

# ------------------------------------------------------------------------------
# Extract TIDigits MFCC features:
# ------------------------------------------------------------------------------
for set in train test; do
    # Create feature and log directories
    mkdir -p $tidigits_feats/logs/make_mfcc/$set
    mkdir -p $tidigits_feats/logs/make_cmvn_dd/$set
    mkdir -p $tidigits_feats/features/mfcc
    # Get raw MFCC features
    /tmp/kaldi/steps/make_mfcc.sh \
        --cmd $train_cmd \
        --mfcc-config $tidigits_feats/conf/mfcc.conf \
        --nj $N_CPU_CORES \
        $tmpdir/$set \
        $tidigits_feats/logs/make_mfcc/$set \
        $tidigits_feats/features/mfcc
    cat $tidigits_feats/features/mfcc/raw_mfcc_$set.*.scp \
        > $tidigits_feats/features/mfcc/raw_mfcc_$set.scp
    rm $tidigits_feats/features/mfcc/raw_mfcc_$set.*.scp
    # Calc cmvn stats
    /tmp/kaldi/steps/compute_cmvn_stats.sh \
        $tmpdir/$set \
        $tidigits_feats/logs/make_mfcc/$set \
        $tidigits_feats/features/mfcc
    # Make MFCC feature with deltas
    $tidigits_feats/cmvn_dd.sh \
        --cmd $train_cmd \
        --nj $N_CPU_CORES \
        $tmpdir/$set \
        $tidigits_feats/logs/make_cmvn_dd/$set \
        $tidigits_feats/features/mfcc
    cat $tidigits_feats/features/mfcc/mfcc_cmvn_dd_$set.*.scp \
        > $tidigits_feats/features/mfcc/mfcc_cmvn_dd_$set.scp
    rm $tidigits_feats/features/mfcc/mfcc_cmvn_dd_$set.*.scp
done

# ------------------------------------------------------------------------------
# Extract TIDigits Filterbank features:
# ------------------------------------------------------------------------------
for set in train test; do
    # Create feature and log directories
    mkdir -p $tidigits_feats/logs/make_fbank/$set
    mkdir -p $tidigits_feats/features/fbank
    # Get Filterbank features
    /tmp/kaldi/steps/make_fbank.sh \
        --compress false \
        --cmd $train_cmd \
        --fbank-config $tidigits_feats/conf/fbank.conf \
        --nj $N_CPU_CORES \
        $tmpdir/$set \
        $tidigits_feats/logs/make_fbank/$set \
        $tidigits_feats/features/fbank
    cat $tidigits_feats/features/fbank/raw_fbank_$set.*.scp \
        > $tidigits_feats/features/fbank/raw_fbank_$set.scp
    rm $tidigits_feats/features/fbank/raw_fbank_$set.*.scp
done

# ------------------------------------------------------------------------------
# Use forced alignments to separate TIDigits sequences into individual digits:
# ------------------------------------------------------------------------------
for set in train test; do
    # Get locations of individual segments found by forced alignment
    $tidigits_feats/tidigits_segments_prep.py \
        $tidigits_feats/forced_align \
        $tmpdir/$set \
        $set
    # Extract MFCCs for the individual digits:
    extract-feature-segments \
        --snip-edges=false \
        --min-segment-length=0.0 \
        "scp:${tidigits_feats}/features/mfcc/mfcc_cmvn_dd_${set}.scp" \
        $tmpdir/$set/segments_indiv \
        "ark,scp:${tidigits_feats}/features/mfcc/mfcc_cmvn_dd_${set}_indiv.ark,${tidigits_feats}/features/mfcc/mfcc_cmvn_dd_${set}_indiv.scp"
    extract-feature-segments \
        --snip-edges=false \
        --min-segment-length=0.0 \
        "scp:${tidigits_feats}/features/fbank/raw_fbank_${set}.scp" \
        $tmpdir/$set/segments_indiv \
        "ark,scp:${tidigits_feats}/features/fbank/raw_fbank_${set}_indiv.ark,${tidigits_feats}/features/fbank/raw_fbank_${set}_indiv.scp"
done

# ------------------------------------------------------------------------------
# Convert the TIDigits Kaldi features to NumPy arrays:
# ------------------------------------------------------------------------------
for set in train test; do
    $tidigits_feats/kaldi_to_numpy.py \
        $tidigits_feats/features/mfcc/mfcc_cmvn_dd_${set}_indiv.scp \
        $tidigits_feats/features/mfcc/mfcc_cmvn_dd_${set}_indiv.npz
    $tidigits_feats/kaldi_to_numpy.py \
        $tidigits_feats/features/fbank/raw_fbank_${set}_indiv.scp \
        $tidigits_feats/features/fbank/raw_fbank_${set}_indiv.npz
done

# ------------------------------------------------------------------------------
# Combine TIDigits features into single numpy archive for easy retrieval:
# ------------------------------------------------------------------------------
$tidigits_feats/combine_archives.py \
    --mfcc-train=$tidigits_feats/features/mfcc/mfcc_cmvn_dd_train_indiv.npz \
    --mfcc-test=$tidigits_feats/features/mfcc/mfcc_cmvn_dd_test_indiv.npz \
    --fbank-train=$tidigits_feats/features/fbank/raw_fbank_train_indiv.npz \
    --fbank-test=$tidigits_feats/features/fbank/raw_fbank_test_indiv.npz \
    --out-file $tidigits_feats/tidigits_audio.npz


# ==============================================================================
#                                  FLICKR AUDIO
# ------------------------------------------------------------------------------
# Prepare feature data directories:
# ------------------------------------------------------------------------------
fa_feats=$FEATURES_DIR/flickr_audio
mkdir -p $fa_feats/features
mkdir -p $fa_feats/logs

# ------------------------------------------------------------------------------
# Prepare Flickr-Audio in the format required for Kaldi feature extraction:
# ------------------------------------------------------------------------------
# Create flickr-audio temporary data directories
tmpdir=/tmp/kaldi/flickr_audio
mkdir -p $tmpdir/train
mkdir -p $tmpdir/test
mkdir -p $tmpdir/local/data
# Make sure that flickr-audio data exists in container at /flickr_audio
rootdir=/flickr_audio  # set tidigits root data directory
if ! [ -d $rootdir ]; then 
    echo "Flickr-Audio directory does not have expected format: ${rootdir}"
    exit 1
fi
# Prepare flickr-audio data
$fa_feats/flickr8k_data_prep_vad.py \
    $rootdir \
    $tmpdir/local/data
# Get duration of complete corpus:
cat $tmpdir/local/data/segments \
    | awk 'BEGIN {sum = 0.0} {sum += $4 - $3} \
           END {printf "Total duration: %.6f hours\n", sum/60.0/60.0}'

# ------------------------------------------------------------------------------
# Extract Flickr-Audio MFCC features for each spoken utterance as a whole:
# ------------------------------------------------------------------------------
# Create feature and log directories
mkdir -p $fa_feats/logs/make_mfcc/full_vad
mkdir -p $fa_feats/logs/make_cmvn_dd/full_vad
mkdir -p $fa_feats/features/mfcc
# Get raw MFCC features
/tmp/kaldi/steps/make_mfcc.sh \
    --cmd $train_cmd \
    --mfcc-config $fa_feats/conf/mfcc.conf \
    --nj $N_CPU_CORES \
    $tmpdir/local/data \
    $fa_feats/logs/make_mfcc/full_vad \
    $fa_feats/features/mfcc
# Calc cmvn stats
/tmp/kaldi/steps/compute_cmvn_stats.sh \
    $tmpdir/local/data \
    $fa_feats/logs/make_mfcc/full_vad \
    $fa_feats/features/mfcc
# Make MFCC feature with deltas
$fa_feats/cmvn_dd.sh \
    --cmd $train_cmd \
    --nj $N_CPU_CORES \
    $tmpdir/local/data \
    $fa_feats/logs/make_cmvn_dd/full_vad \
    $fa_feats/features/mfcc
cat $fa_feats/features/mfcc/mfcc_cmnv_dd_data.*.scp \
    > $fa_feats/features/mfcc/mfcc_cmnv_dd_full_vad.scp
rm $fa_feats/features/mfcc/mfcc_cmnv_dd_data.*.scp

# ------------------------------------------------------------------------------
# Extract Flickr-Audio Filterbank features for each spoken utterance as a whole:
# ------------------------------------------------------------------------------
# Create feature and log directories
mkdir -p $fa_feats/logs/make_fbank/$set
mkdir -p $fa_feats/features/fbank
# Get Filterbank features
/tmp/kaldi/steps/make_fbank.sh \
    --compress false \
    --cmd $train_cmd \
    --fbank-config $fa_feats/conf/fbank.conf \
    --nj $N_CPU_CORES \
    $tmpdir/local/data \
    $fa_feats/logs/make_fbank/full_vad \
    $fa_feats/features/fbank
cat $fa_feats/features/fbank/raw_fbank_data.*.scp \
    > $fa_feats/features/fbank/raw_fbank_full_vad.scp
rm $fa_feats/features/fbank/raw_fbank_data.*.scp

# ------------------------------------------------------------------------------
# Convert to NumPy archives with train-test-dev splits of Flickr-8k-Text corpus:
# ------------------------------------------------------------------------------
$fa_feats/get_kaldi_fbank.py \
    /Flickr8k_text \
    $fa_feats/features/fbank \
    raw_fbank_full_vad.scp
$fa_feats/get_kaldi_mfcc.py \
    /Flickr8k_text \
    $fa_feats/features/mfcc \
    mfcc_cmnv_dd_full_vad.scp

# ------------------------------------------------------------------------------
# Split each subset into isolated words:
# ------------------------------------------------------------------------------
for set in dev train test; do
    $fa_feats/get_iso_words.py \
        $rootdir \
        $fa_feats/features \
        mfcc \
        $set
    $fa_feats/get_iso_words.py \
        $rootdir \
        $fa_feats/features \
        fbank \
        $set
done

# ------------------------------------------------------------------------------
# Combine Flickr-Audio features into single numpy archive for easy retrieval:
# ------------------------------------------------------------------------------
$fa_feats/combine_archives.py \
    --mfcc-train=$fa_feats/features/mfcc/train_words.npz \
    --mfcc-dev=$fa_feats/features/mfcc/dev_words.npz \
    --mfcc-test=$fa_feats/features/mfcc/test_words.npz \
    --fbank-train=$fa_feats/features/fbank/train_words.npz \
    --fbank-dev=$fa_feats/features/fbank/dev_words.npz \
    --fbank-test=$fa_feats/features/fbank/test_words.npz \
    --out-file $fa_feats/flickr_audio.npz
