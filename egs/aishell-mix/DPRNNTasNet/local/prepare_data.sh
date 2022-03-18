#!/bin/bash

storage_dir=
n_src=
python_path=python

. ./utils/parse_options.sh

if [[ $n_src -le  1 ]]
then
  changed_n_src=2
else
  changed_n_src=$n_src
fi

$python_path local/create_local_metadata.py --aishellmix_dir $storage_dir/Aishell${changed_n_src}"Mix"

$python_path local/get_text.py \
  --aishelldir $storage_dir/data_aishell/wav \
  --split test \
  --outfile data/test_annotations.csv
