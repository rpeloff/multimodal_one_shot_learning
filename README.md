Cross-Modal Few-Shot Learning
=============================

Overview
--------
Few-shot learning experiments on cross-modal vision-speech retrieval, where a given query in *one* modality is used to match an example with the most similar content from a (test) set of possible answers that are in *another modality*. This cross-modal matching is performed based on knowledge of another set of matching vision-speech examples (known as an acquisition set), where only one or a few matching examples are available per class.

Datasets
--------

Need to obtain the following datasets:

TIDigits: https://catalog.ldc.upenn.edu/LDC93S10
Flickr audio: https://groups.csail.mit.edu/sls/downloads/flickraudio/
Flickr8k text: http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip

Note that the Flickr8k text corpus is used purely for obatining the train/validation/test splits.
The following instrcutions assume that you have obtained these datasets and placed them somewhere sensible (e.g. ../data/tidigits)

Pre-requisites
--------------

Install docker: https://docs.docker.com/engine/installation/
Install nvidia-docker for GPU access in docker containers (version 2.x I think): https://github.com/NVIDIA/nvidia-docker

Pull docker images from Docker Hub:
1. Kaldi docker image for extracting speech features https://hub.docker.com/r/reloff/kaldi/:
`docker pull reloff/kaldi`

2. TensorFlow docker image used as base image for research environment https://hub.docker.com/r/reloff/tensorflow-base/:
`docker pull reloff/tensorflow-base`

3. Multimodal one-shot research environment docker image https://hub.docker.com/r/reloff/multimodal-one-shot/:
`docker pull reloff/multimodal-one-shot`

Alternatively you can build these images locally from their DockerFiles:
1. https://github.com/rpeloff/research-images/blob/master/kaldi/Dockerfile 
2. https://github.com/rpeloff/research-images/blob/master/tensorflow_base/python36_cuda90/Dockerfile
3. https://github.com/rpeloff/multimodal-one-shot-learning/blob/master/docker/Dockerfile

Kaldi feature extraction
------------------------

Extract speech features by simply running:

`./run_feature_extraction.sh \
    --tidigits=<path to TIDigits> \
    --flickr-audio=<path to Flickr audio> \
    --flickr-text=<path to Flickr8k text> \
    --n-cpu-cores=<number of CPU cores>`
    
Replace each path with the full path to the corresponding dataset (e.g. `--flickr-audio=/home/rpeloff/datasets/speech/flickr_audio`).
The `--n-cpu-cores` flag specifies the number of CPU cores to use to speed up feature extraction (defaults to 8; set higher or lower depending on available CPU cores).

