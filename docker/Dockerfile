# ------------------------------------------------------------------------------
# Docker image for multimodal one-shot learning research.
#
# Author: Ryan Eloff
# Contact: ryan.peter.eloff@gmail.com
# Date: September 2018
# ------------------------------------------------------------------------------
# Based on the TensorFlow image from https://github.com/rpeloff/research-images.
# Usage with nvidia-docker:
#   $ nvidia-docker run -it --name my_container <image-build-tag> bash
# ------------------------------------------------------------------------------
FROM reloff/tensorflow-base:1.11.0-py36-cuda90
LABEL author="Ryan Eloff" date="28-09-2018"
# ------------------------------------------------------------------------------
# Install requirements from requirements.txt
# ------------------------------------------------------------------------------
COPY requirements.txt /tmp
RUN python -m pip --no-cache-dir install --upgrade -r /tmp/requirements.txt \
&& rm -rf /tmp/*
# ------------------------------------------------------------------------------
# Set default command to '/bin/sh -c bash'
# ------------------------------------------------------------------------------
CMD ["bash"]
