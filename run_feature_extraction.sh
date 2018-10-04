#!/bin/bash

# Extract TIDigits and Flickr-Audio features in a Kaldi Docker container.
#
# Author: Ryan Eloff
# Contact: ryan.peter.eloff@gmail.com
# Date: August 2018


# Process keyword arguments
print_help_and_exit=false
while getopts ":-:h" opt; do
    # Check option not specified without arg (if arg is required; takes next option "-*" as arg)
    if [[ ${OPTARG} == -* ]]; then
        echo "Option -${opt} requires an argument." >&2 && print_help_and_exit=true
    fi
    # Handle specified option and arguments
    case $opt in
        h)  # Print help
            print_help_and_exit=true
            ;;
        :)  # Options specified without any arg evaluate to ":" (if arg is required)
            echo "Option -$OPTARG requires an argument." >&2 && print_help_and_exit=true
            ;;
        -)  # long options (--long-option arg)
            case $OPTARG in
                tidigits|tidigits=*)
                    val=${OPTARG#tidigits}  # remove "tidigits" from the opt arg
                    val=${val#*=}  # get value after "="
                    opt=${OPTARG%=$val}  # get option before "=value" (debug)
                    if [ -z "${val}" ]; then  # check value is not empty
                        echo "Option --${opt} is missing the directory argument!" >&2 && print_help_and_exit=true
                    else
                        TIDIGITS_DIR=$val
                    fi
                    ;;
                flickr-audio|flickr-audio=*)
                    val=${OPTARG#flickr-audio}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the directory argument!" >&2 && print_help_and_exit=true
                    else
                        FLICKR_DIR=$val
                    fi
                    ;;
                n-cpu-cores|n-cpu-cores=*)
                    val=${OPTARG#n-cpu-cores}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the number argument!" >&2 && print_help_and_exit=true
                    else
                        N_CPU_CORES=$val
                    fi
                    ;;
                name|name=*)
                    val=${OPTARG#name}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the name argument!" >&2 && print_help_and_exit=true
                    else
                        DOCKER_NAME=$val
                    fi
                    ;;
		        help)
		            print_help_and_exit=true
		            ;;
                *)
                    echo "Invalid option --${OPTARG}" >&2 && print_help_and_exit=true
                    ;;
            esac
            ;;
        \?)
            echo "Invalid option -${opt} ${OPTARG}" >&2 && print_help_and_exit=true
            ;;
    esac
done


# Print help and exit for invalid input or -h/--help option
if [ "${print_help_and_exit}" = true ]; then
    echo ""
    echo "Usage: run_feature_extraction.sh [OPTIONS]"
    echo ""
    echo "Extract TIDigits and Flickr-Audio features in a Kaldi Docker container"
    echo ""
    echo "Options:"
    echo "        --tidigits dir        Path to the TIDigits data."
    echo "        --flickr-audio dir    Path to the Flickr-Audio data."
    echo "        --n-cpu-cores         Number of CPU cores to use during feature extraction."
    echo "        --name string         Name for the Docker container (Default: kaldi-feats-extract)."
    echo "    -h, --help                Print this information and exit."
    echo ""
    exit 1
fi


# Set the location of the TIDigits dataset (default dir used as example)
TIDIGITS_DIR=${TIDIGITS_DIR:-/home/rpeloff/datasets/speech/tidigits}
# Set the location of the Flickr-Audio dataset (default dir used as example)
FLICKR_DIR=${FLICKR_DIR:-/home/rpeloff/datasets/speech/flickr_audio}
# Set number of CPU cores to use during feature extraction (default: 8)
N_CPU_CORES=${N_CPU_CORES:-8}
# Set name of docker container (default name: kaldi-feats-extract)
DOCKER_NAME=${DOCKER_NAME:-kaldi-feats-extract}


# Print some information on selected options
echo "Starting experiment notebook container!"
echo ""
echo "TIDigits directory: ${TIDIGITS_DIR}"
echo "Flickr-Audio directory: ${FLICKR_DIR}"
echo "Number of CPU cores: ${N_CPU_CORES}"
echo "Docker container name: ${DOCKER_NAME}"
echo ""


# Start Docker feature extraction container
# Note: run script as sudo if Docker not set up for non-root user
docker run \
    -v ${TIDIGITS_DIR}:/tidigits \
    -v ${FLICKR_DIR}:/flickr_audio \
    -v `pwd`/kaldi_features:/kaldi_features \
    -e FEATURES_DIR=/kaldi_features \
    -e N_CPU_CORES=${N_CPU_CORES} \
    --rm \
    -it \
    --name ${DOCKER_NAME} \
    reloff/kaldi:5.4 \
    /kaldi_features/extract_features.sh
