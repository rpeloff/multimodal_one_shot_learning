Cross-Modal Few-Shot Learning
=============================

Overview
--------
Few-shot learning experiments on cross-modal vision-speech retrieval, where a given query in *one* modality is used to match an example with the most similar content from a (test) set of possible answers that are in *another modality*. This cross-modal matching is performed based on knowledge of another set of matching vision-speech examples (known as an acquisition set), where only one or a few matching examples are available per class.

Datasets
--------

The following datasets are required for these experiments:

- [TIDigits](https://catalog.ldc.upenn.edu/LDC93S10)
- [Flickr audio](https://groups.csail.mit.edu/sls/downloads/flickraudio/)
- [Flickr8k text](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip)

Note that the Flickr8k text corpus is used purely for obtaining train/validation/test splits.
The instructions that follow assume that you have obtained these datasets and placed them somewhere sensible (e.g. ../data/tidigits)

Pre-requisites
--------------
The following steps need to be completed to run the experiment scripts:

1. Install [Docker](https://docs.docker.com/install/) (I also recommend following the [linux post-install step](https://docs.docker.com/install/linux/linux-postinstall/) to manage Docker as a non-root user)

2. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (version 2.0) for NVIDIA GPU access in docker containers

3. Pull required images from [Docker Hub](https://hub.docker.com):

| Docker image | Docker pull command |
| ------------- | -------------------|
| [Kaldi](https://hub.docker.com/r/reloff/kaldi) for extracting speech features | `docker pull reloff/kaldi:5.4` |
| [TensorFlow](https://hub.docker.com/r/reloff/tensorflow-base) used as base for research environment | `docker pull reloff/tensorflow-base:1.11.0-py36-cuda90` |
| [Multimodal one-shot research environment](https://hub.docker.com/r/reloff/multimodal-one-shot) | `docker pull reloff/multimodal-one-shot` |

Alternatively you can build these images locally from their DockerFiles:
- [Kaldi Dockerfile](https://github.com/rpeloff/research-images/blob/master/kaldi/Dockerfile)
- [TensorFlow Dockerfile](https://github.com/rpeloff/research-images/blob/master/tensorflow_base/python36_cuda90/Dockerfile)
- [Multimodal one-shot research environment Dockerfile](https://github.com/rpeloff/multimodal-one-shot-learning/blob/master/docker/Dockerfile)

Kaldi feature extraction
------------------------

Extract speech features by simply running:

```bash
./run_feature_extraction.sh \
    --tidigits=<path to TIDigits> \
    --flickr-audio=<path to Flickr audio> \
    --flickr-text=<path to Flickr8k text> \
    --n-cpu-cores=<number of CPU cores>
```
    
Replace each path with the full path to the corresponding dataset (e.g. `--flickr-audio=/home/rpeloff/datasets/speech/flickr_audio`).
The `--n-cpu-cores` flag specifies the number of CPU cores used for feature extraction (defaults to 8; set higher or lower depending on available CPU cores), where more cores may speed up the process.

