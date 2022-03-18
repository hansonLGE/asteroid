#!/bin/bash

storage_dir=
n_src=
python_path=python

. ./utils/parse_options.sh

current_dir=$(pwd)

# Run generation script
cd AishellMix
. generate_aishellmix.sh $storage_dir
