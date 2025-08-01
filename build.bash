#!/bin/bash

# ================================= Edit Here ================================ #

# TODO: Change these values to use different versions of ROS or different base images. The rest of the script should be left unchanged.
#       Remember to change the ROS_NUMBER in accordance to the ROS version, otherwise you're stupid and deserve a painful death.
BASE_IMAGE=osrf/ros
BASE_TAG=humble-desktop
ROS_NUMBER=2
IMAGE_NAME=ho_mpc

# =============================== Preliminaries ============================== #

mkdir -p build log
if [[ $ROS_NUMBER == 1 ]]; then
    mkdir -p devel
elif [[ $ROS_NUMBER == 2 ]]; then
    mkdir -p install
fi

mkdir -p ~/.vscode ~/.vscode-server ~/.config/Code

# =============================== Help Function ============================== #

helpFunction()
{
    echo ""
    echo "Usage: $0 [-a] [-f] [-l] [-r]"
    echo -e "\t-a   --all       Install everything."
    echo -e "\t-f   --ffmpeg    Install ffmpeg to save videos."
    echo -e "\t-h   --help      Print the help."
    echo -e "\t-l   --latex     Install latex for nicer images with Matplotlib."
    echo -e "\t-r   --rebuild   Rebuild the image."
    echo -e "\t-t   --torch     Install PyTorch and related packages."
    exit 1 # Exit script after printing help
}

# =============================== Build Options ============================== #

# Initialie the build options
LATEX=0
FFMPEG=0
REBUILD=0

# Auxiliary functions
die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }
no_arg() { if [ -n "$OPTARG" ]; then die "No arg allowed for --$OPT option"; fi; }

# Get the script options. This accepts both single dash (e.g. -a) and double dash options (e.g. --all)
while getopts afhlr-: OPT; do
    # support long options: https://stackoverflow.com/a/28466267/519360
    if [ "$OPT" = "-" ]; then     # long option: reformulate OPT and OPTARG
        OPT="${OPTARG%%=*}"       # extract long option name
        OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
        OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
    fi
    case "$OPT" in
        a | all )       no_arg; FFMPEG=1 LATEX=1 TORCH=1;;
        f | ffmpeg )    no_arg; FFMPEG=1 ;;
        h | help )      no_arg; helpFunction ;;
        l | latex )     no_arg; LATEX=1 ;;
        r | rebuild )   no_arg; REBUILD=1 ;;
        t | torch )     no_arg; TORCH=1 ;;
        ??* )           die "Illegal option --$OPT" ;;  # bad long option
        ? )             exit 2 ;;  # bad short option (error reported via getopts)
    esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

# ========================= Pull And Build The Image ========================= #

docker pull $BASE_IMAGE:$BASE_TAG

MYUID="$(id -u $USER)"
MYGID="$(id -g $USER)"

if [ "$REBUILD" -eq 1 ]; then
    cache="--no-cache"
else
    cache=""
fi

docker build \
    ${cache} \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    --build-arg BASE_TAG=$BASE_TAG \
    --build-arg ROS_NUMBER=$ROS_NUMBER \
    --build-arg MYUID=${UID} \
    --build-arg MYGID=${GID} \
    --build-arg USER=${USER} \
    --build-arg "PWDR=$PWD" \
    --build-arg FFMPEG=$FFMPEG \
    --build-arg LATEX=$LATEX \
    --build-arg TORCH=$TORCH \
    -t $IMAGE_NAME .
