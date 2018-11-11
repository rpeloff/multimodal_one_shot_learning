#!/bin/bash

# Run experiment Notebooks in the Docker research container.
#
# Author: Ryan Eloff
# Contact: ryan.peter.eloff@gmail.com
# Date: August 2018


# Process keyword arguments
print_help_and_exit=false
while getopts ":p:-:h" opt; do
    # Check option not specified without arg (if arg is required; takes next option "-*" as arg)
    if [[ ${OPTARG} == -* ]]; then
        echo "Option -${opt} requires an argument." >&2 && print_help_and_exit=true
    fi
    # Handle specified option and arguments
    case $opt in
        h)  # print help
            print_help_and_exit=true
            ;;
        p)  # set Jupyter notebook port
            JUP_PORT="$OPTARG"
            ;;
        :)  # options specified without any arg evaluate to ":" (if arg is required)
            echo "Option -$OPTARG requires an argument." >&2 && print_help_and_exit=true
            ;;
        -)  # long options (--long-option arg)
            case $OPTARG in
                port|port=*)
                    val=${OPTARG#port}  # remove "port" from the opt arg
                    val=${val#*=}  # get value after "="
		            opt=${OPTARG%=$val}  # get option before "=value" (debug)
                    if [ -z "${val}" ]; then  # check value is not empty
                        echo "Option --${opt} is missing the port number argument!" >&2 && print_help_and_exit=true
                    else
                        JUP_PORT=$val  # assign port value
                    fi
		            ;;
                image|image=*)
                    val=${OPTARG#image}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the image argument!" >&2 && print_help_and_exit=true
                    else
                        DOCKER_IMAGE=$val
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
                password|password=*)
                    val=${OPTARG#password}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the password argument!" >&2 && print_help_and_exit=true
                    else
                        JUP_PASS=$val
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
    echo "Usage: run_notebooks.sh [OPTIONS]"
    echo ""
    echo "Run experiment Notebooks in the Docker research container"
    echo "(Run script with sudo if Docker is not set up for non-root users)"
    echo ""
    echo "Options:"
    echo "    -p, --port number         Port the Jupyter notebook server will listen on (Default: 8888)."
    echo "        --image string        Image for the Docker container (Default: reloff/multimodal-one-shot)."
    echo "        --name string         Name for the Docker container (Default: multimodal-one-shot)."
    echo "        --password string     Token used to access the Jupyter notebook (Default: No password)."
    echo "    -h, --help                Print this information and exit."
    echo ""
    exit 1
fi


# Set local notebook port (default port: 8888)
JUP_PORT=${JUP_PORT:-8888}
# Set image for docker container (default image: reloff/multimodal-one-shot)
DOCKER_IMAGE=${DOCKER_IMAGE:-reloff/multimodal-one-shot}
# Set name of docker container (default name: multimodal-one-shot)
DOCKER_NAME=${DOCKER_NAME:-multimodal-one-shot}
# Set password of notebook (default password: '' - no authentication)
JUP_PASS=${JUP_PASS:-''}


# Print some information on selected options
echo "Starting experiment notebook container!"
echo "(Run script with sudo if Docker is not set up for non-root users)"
echo ""
echo "Jupyter notebook port: ${JUP_PORT}"
echo "Docker container image: ${DOCKER_IMAGE}"
echo "Docker container name: ${DOCKER_NAME}"
echo "Jupyter notebook access token: ${JUP_PASS}"
echo ""


# Start Docker research container
# Note: run script as sudo if Docker not set up for non-root user
nvidia-docker run \
    -v $(pwd):/multimodal_one_shot \
    -u $(id -u):$(id -g) \
    -w /multimodal_one_shot \
    --rm \
    -it \
    -e JUPYTER_DATA_DIR=/multimodal_one_shot/.jupyter \
    -p ${JUP_PORT}:${JUP_PORT} \
    --name ${DOCKER_NAME} \
    ${DOCKER_IMAGE} \
    jupyter notebook --no-browser --ip=0.0.0.0 --port=${JUP_PORT} --NotebookApp.token=${JUP_PASS} --notebook-dir='/multimodal_one_shot/experiments'
