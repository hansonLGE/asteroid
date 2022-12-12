#!/bin/bash

# Exit on error
set -e
set -o pipefail

# If you haven't generated AishellMix start from stage 0
# Main storage directory. You'll need disk space to store Aishell, WHAM noises
# and AishellMix. This is about 400 Gb
storage_dir=/server/speech_data/Mandarin/aishell

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=0  # Controls from which stage to start
stop_stage=3
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES
out_dir=aishellmix # Controls the directory name associated to the evaluation results inside the experiment directory

# Network config

# Training config
epochs=200
batch_size=4
num_workers=8
half_lr=yes
early_stop=yes
# Optim config
optimizer=adam
lr=0.001
weight_decay=0.00001
# Data config
sample_rate=16000
mode=min
n_src=2
segment=3
task=sep_clean  # one of 'enh_single', 'enh_both', 'sep_clean', 'sep_noisy'

# Architecture config
kernel_size=16
stride=8
chunk_size=100
hop_size=50

# Evaluation
eval_use_gpu=1
# Need to --compute_wer 1 --eval_mode max to be sure the user knows all the metrics
# are for the all mode.
compute_wer=0
eval_mode=

. utils/parse_options.sh


sr_string=$(($sample_rate/1000))
suffix=wav${sr_string}k/$mode

if [ -z "$eval_mode" ]; then
  eval_mode=$mode
fi

train_dir=data/$suffix/train
valid_dir=data/$suffix/dev
test_dir=data/wav${sr_string}k/$eval_mode/test

if [ $stage -le  0 ] && [ ${stop_stage} -ge 0 ]; then
	echo "Stage 0: Generating Aishellmix dataset"
  . local/generate_aishellmix.sh --storage_dir $storage_dir --n_src $n_src
fi

if [ $stage -le  1 ] && [ ${stop_stage} -ge 1 ]; then
	echo "Stage 1: Generating csv files including wav path and duration"
  . local/prepare_data.sh --storage_dir $storage_dir --n_src $n_src
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi

expdir=exp/train_dprnntasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py --exp_dir $expdir \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--half_lr $half_lr \
		--early_stop $early_stop \
		--optimizer $optimizer \
		--lr $lr \
		--weight_decay $weight_decay \
		--train_dir $train_dir \
		--valid_dir $valid_dir \
		--sample_rate $sample_rate \
		--n_src $n_src \
		--task $task \
                --kernel_size $kernel_size \
                --stride $stride \
                --chunk_size $chunk_size \
                --hop_size $hop_size \
                --segment $segment | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "aishellmix/DPRNNTasNet" > $expdir/publish_dir/recipe_name.txt
fi


if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3 : Evaluation"

  if [[ $compute_wer -eq 1 ]]; then
    if [[ $eval_mode != "max" ]]; then
      echo "Cannot compute WER without max mode. Start again with --stage 2 --compute_wer 1 --eval_mode max"
      exit 1
    fi

    # Install espnet if not instaled
    if ! python -c "import espnet" &> /dev/null; then
        echo 'This recipe requires espnet. Installing requirements.'
        $python_path -m pip install espnet_model_zoo
        $python_path -m pip install jiwer
        $python_path -m pip install tabulate
    fi
  fi

  $python_path eval.py \
    --exp_dir $expdir \
    --test_dir $test_dir \
  	--out_dir $out_dir \
  	--use_gpu $eval_use_gpu \
  	--compute_wer $compute_wer \
  	--task $task | tee logs/eval_${tag}.log

	cp logs/eval_${tag}.log $expdir/eval.log
fi
