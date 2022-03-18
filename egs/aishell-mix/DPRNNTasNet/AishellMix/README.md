### About the dataset
AishellMix is an open source dataset for source separation in noisy 
environments. It is derived from Aishell-1 signals and WHAM noise. 
It offers a free alternative to the WHAM dataset and complements it. 
It will also enable cross-dataset experiments.

### Genetating AishellMix
To generate wham noise metadata, run the script:
```
python scripts/create_wham_metadata.py --wham_dir /server/speech_data/librispeech/wham_noise
```
move the files:train.csv dev.csv test.csv to current directory metadata/Wham_noise

To generate aishell metadata, run the script:
```
python scripts/create_aishell_metadata.py --aishell_dir /server/speech_data/Mandarin/aishel
```
move the files:train.csv dev.csv test.csv to current directory metadata/Aishell

To generate aishellmix metadata, run the script:
```
python scripts/create_aishellmix_metadata.py --aishell_dir /server/speech_data/Mandarin/aishell/data_aishell/wav --aishell_md_dir /server/speech_data/project/asteroid/egs/aishell-mix/ConvTasNet/AishellMix/metadata/Aishell --wham_dir /server/speech_data/librispeech/wham_noise --wham_md_dir /server/speech_data/project/asteroid/egs/aishell-mix/ConvTasNet/AishellMix/metadata/Wham_noise --metadata_outdir /server/speech_data/Mandarin/aishell --n_src 2
```
 
### Generating AishellMix
To generate AishellMix, run the main script : 
[`generate_aishellmix.sh`](./generate_aishellmix.sh)

```
cd AishellMix 
./generate_aishellmix.sh storage_dir
```

Make sure that SoX is installed on your machine.

For windows :
```
conda install -c groakat sox
```

For Linux :
```
conda install -c conda-forge sox
```

You can either change `storage_dir` and `n_src` by hand in 
the script or use the command line.  
By default, AishellMix will be generated for 2 and 3 speakers,
at both 16Khz and 8kHz, 
for min max modes, and all mixture types will be saved (mix_clean, 
mix_both and mix_single). 


### Features
In AishellMix you can choose :
* The number of sources in the mixtures.
* The sample rate  of the dataset from 16 KHz to any frequency below. 
* The mode of mixtures : min (the mixture ends when the shortest source
 ends) or max (the mixtures ends with the longest source)
 * The type of mixture : mix_clean (utterances only) mix_both (utterances + noise) mix_single (1 utterance + noise)

You can customize the generation by editing ``` generate_aishellmix.sh ```.
 
### Note on scripts
For the sake of transparency, we have released the metadata generation 
scripts. However, we wish to avoid any changes to the dataset, 
especially to the test subset that shouldn't be changed under any 
circumstance.
