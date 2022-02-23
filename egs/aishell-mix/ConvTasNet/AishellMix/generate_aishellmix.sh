#!/bin/bash
set -eu  # Exit on error

storage_dir=$1
aishell_dir=$storage_dir/data_aishell/wav
#wham_dir=$storage_dir/wham_noise
wham_dir=/server/speech_data/librispeech/wham_noise
aishellmix_outdir=$storage_dir/

function Aishell_1 {
	if ! test -e $aishell_dir; then
		echo "Download Aishell_1 into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 https://us.openslr.org/resources/33/data_aishell.tgz -P $storage_dir
                wget -c --tries=0 --read-timeout=20 https://us.openslr.org/resources/33/resource_aishell.tgz -P $storage_dir
		tar -xzf $storage_dir/data_aishell.tgz -C $storage_dir
                tar -xzf $storage_dir/resource_aishell.tgz -C $storage_dir
		rm -rf $storage_dir/data_aishell.tgz
                rm -rf $storage_dir/resource_aishell.tgz
	fi
}

function wham() {
	if ! test -e $wham_dir; then
		echo "Download wham_noise into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 https://storage.googleapis.com/whisper-public/wham_noise.zip -P $storage_dir
		unzip -qn $storage_dir/wham_noise.zip -d $storage_dir
		rm -rf $storage_dir/wham_noise.zip
	fi
}

#Aishell_1 &
#wham &

#wait

# Path to python
python_path=python

# If you wish to rerun this script in the future please comment this line out.
$python_path scripts/augment_train_noise.py --wham_dir $wham_dir

for n_src in 2 3; do
  metadata_dir=metadata/Aishell$n_src"Mix"
  $python_path scripts/create_aishellmix_from_metadata.py --aishell_dir $aishell_dir \
    --wham_dir $wham_dir \
    --metadata_dir $metadata_dir \
    --aishellmix_outdir $aishellmix_outdir \
    --n_src $n_src \
    --freqs 8k 16k \
    --modes min max \
    --types mix_clean mix_both mix_single
done
