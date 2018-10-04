#!/bin/bash
# Herman Kamper, kamperh@gmail.com, 2015.
# Based loosely an parts of train_mono.sh.

nj=4
cmd=run.pl

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

echo "CMVN_DD.sh: -nj" ${nj} "--cmd" ${cmd}

if [ $# != 3 ]; then
    echo "usage: ${0} data_dir exp_dir feat_dir"
    exit 1;
fi

data=$1
logdir=$2
mfccdir=$3

dataname=`basename $data`

mkdir -p $logdir/log
echo $nj > $logdir/num_jobs
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

feats="apply-cmvn --norm-vars=true --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark,scp:$mfccdir/mfcc_cmvn_dd_$dataname.JOB.ark,$mfccdir/mfcc_cmvn_dd_$dataname.JOB.scp"

$train_cmd JOB=1:$nj $logdir/log/cmvn_dd.JOB.log $feats || exit 1;
