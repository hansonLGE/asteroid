import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
import random as random
import os
import glob
import time

from tqdm import tqdm
from pysndfx import AudioEffectsChain


class AugmentedDataset(Dataset):
    """Dataset class for dynamic mixing to source separation tasks.

    Args:
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'`` :

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (int, optional) : The desired sources and mixtures length in s.
        aug_train_dir (str): The path to the directory containing the augmented train/dev/test .wav files.
        noise_dir (str, optional): The path to the directory containing the WHAM train/dev/test .wav files.
        global_db_range: (tuple, optional): Minimum and maximum bounds for each source (and noise) (dB).
        global_stats: (tuple, optional): Mean and standard deviation for level in dB of first source.
        rel_stats: (tuple, optional): Mean and standard deviation for level in dB of second source relative to the first source.
        noise_stats: (tuple, optional): Mean and standard deviation for level in dB of noise relative to the first source.
        speed_perturb: (tuple, optional): Range for SoX speed perturbation transformation.

    """


    def __init__(
        self,
        aug_train_dir,
        task="sep_clean",
        sample_rate=16000,
        n_src=2,
        segment=3,
        return_id=False,
        noise_dir=None,
        global_db_range=(-45, 0),
        abs_stats=(-16.7, 7),
        rel_stats=(2.52, 4),
        noise_stats=(5.1, 6.4),
        speed_perturb=(0.95, 1.05)
    ):
        self.task = task
        if self.task in ["sep_noisy", "enh_single"] and not noise_dir:
            raise RuntimeError(
                "noise directory must be specified if task is sep_noisy or enh_single"
            )
        self.return_id = return_id
        self.segment = segment
        self.sample_rate = sample_rate
        self.seg_len = None if segment is None else int(segment * sample_rate)
        self.n_src = n_src
        self.global_db_range = global_db_range
        self.abs_stats = abs_stats
        self.rel_stats = rel_stats
        self.noise_stats = noise_stats
        self.speed_perturb = speed_perturb

        self.hashtab_synth = self.parse_augmented(aug_train_dir, noise_dir)
		
    def parse_augmented(self, aug_train_dir, noise_dir):
        # Recursively look for .wav files in current directory
        utterances = glob.glob(os.path.join(aug_train_dir, "**/*.wav"), recursive=True)
        noises = None
        if self.task in ["sep_noisy", "enh_single", "enh_both"]:
            noises = glob.glob(os.path.join(noise_dir, "*.wav"))
            assert len(noises) > 0, "No noises parsed. Wrong path?"

        # parse utterances according to speaker
        drop_utt, drop_len = 0, 0
        print("Parsing Aishell-1 speakers")
        examples_hashtab = {}
        # Go through the sound file list
        for utt in tqdm(utterances, total=len(utterances)):
            # exclude if too short
            meta = sf.SoundFile(utt)
            c_len = len(meta)
            assert meta.samplerate == self.sample_rate

            target_length = (
                int(np.ceil(self.speed_perturb[1] * self.seg_len))
                if self.speed_perturb
                else self.seg_len
            )

            if c_len < target_length:  # speed perturbation
                drop_utt += 1
                drop_len += c_len
                continue

            speaker = utt.split("/")[-2]  # Get the ID of the speaker
            if speaker not in examples_hashtab.keys():
                examples_hashtab[speaker] = [(utt, c_len)]
            else:
                examples_hashtab[speaker].append((utt, c_len))

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / self.sample_rate / 36000, len(utterances), target_length
            )
        )

        drop_utt, drop_len = 0, 0
        if noises:
            examples_hashtab["noise"] = []
            for noise in tqdm(noises, total=len(noises)):
                meta = sf.SoundFile(noise)
                c_len = len(meta)
                assert meta.samplerate == self.sample_rate
                target_length = (
                    int(np.ceil(self.speed_perturb[1] * self.seg_len))
                    if self.speed_perturb
                    else self.seg_len
                )
                if c_len < target_length:  # speed perturbation
                    drop_utt += 1
                    drop_len += c_len
                    continue
                examples_hashtab["noise"].append((noise, c_len))

            print(
                "Drop {} noises({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / self.sample_rate / 36000, len(noises), self.seg_len
                )
            )

        return examples_hashtab

    def __len__(self):
        size = sum(
            [len(self.hashtab_synth[x]) for x in self.hashtab_synth.keys()]
        )  # we account only the Augmented data length

        return size

    def random_data_augmentation(self, signal, c_gain, speed):
        if self.speed_perturb:
            fx = (
                AudioEffectsChain().speed(speed).custom("norm {}".format(c_gain))
            )  # speed perturb and then apply gain
        else:
            fx = AudioEffectsChain().custom("norm {}".format(c_gain))
        signal = fx(signal)

        return signal

    @staticmethod
    def get_random_subsegment(array, desired_len, tot_length):

        offset = 0
        if desired_len < tot_length:
            offset = np.random.randint(0, tot_length - desired_len)

        out, _ = sf.read(array, start=offset, stop=offset + desired_len, dtype="float32")

        if len(out.shape) > 1:
            out = out[:, random.randint(0, 1)]

        return out

    def write_soundfile(self, file_id, mixture, dir_path, subdir, freq):
        # Write noise save it's path
        ex_filename = file_id + '.wav'
        save_path = os.path.join(dir_path, subdir, ex_filename)
        abs_save_path = os.path.abspath(save_path)
        sf.write(abs_save_path, mixture, freq)
        return abs_save_path

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # return augmented data: Sample k speakers randomly
        c_speakers = np.random.choice(
            [x for x in self.hashtab_synth.keys() if x != "noise"], self.n_src
        )

        sources = []
        ids = []
        first_lvl = None
        floor, ceil = self.global_db_range
        for i, spk in enumerate(c_speakers):
            c_speed = 0
            tmp, tmp_spk_len = random.choice(self.hashtab_synth[c_speakers[i]])
            # BAC009S0002W0122.wav
            source_id = tmp.split("/")[-1].split(".")[0]
            # account for sample reduction in speed perturb
            if self.speed_perturb:
                c_speed = round(random.uniform(*self.speed_perturb), 2)
                target_len = int(np.ceil(c_speed * self.seg_len))
            else:
                target_len = self.seg_len
            ids.append(source_id + 'speed' + str(c_speed))
            tmp = self.get_random_subsegment(tmp, target_len, tmp_spk_len)

            # Write orgi source segment for check
            #dir_path ='/server/speech_data/Mandarin/aishell'
            #subdir = '0316check'
            #print("write orgi file: {}".format(source_id))
            #self.write_soundfile(source_id, tmp, dir_path, subdir, 16000)

            if i == 0:  # we model the signal level distributions with gaussians
                c_lvl = np.clip(random.normalvariate(*self.abs_stats), floor, ceil)
                first_lvl = c_lvl
            else:
                c_lvl = np.clip(first_lvl - random.normalvariate(*self.rel_stats), floor, ceil)
            tmp = self.random_data_augmentation(tmp, c_lvl, c_speed)
            tmp = tmp[: self.seg_len]

            # Write speed perturb source segment for check
            #print("write speed file: {}".format(source_id + 'speed' + str(c_speed)))
            #self.write_soundfile(source_id + 'speed' + str(c_speed), tmp, dir_path, subdir, 16000)

            sources.append(tmp)

        if self.task in ["sep_noisy", "enh_single", "enh_both"]:
            # add also noise
            tmp, tmp_spk_len = random.choice(self.hashtab_synth["noise"])
            if self.speed_perturb:
                c_speed = round(random.uniform(*self.speed_perturb), 2)
                target_len = int(np.ceil(c_speed * self.seg_len))
            else:
                target_len = self.seg_len
            tmp = self.get_random_subsegment(tmp, target_len, tmp_spk_len)
            c_lvl = np.clip(first_lvl - random.normalvariate(*self.noise_stats), floor, ceil)
            tmp = self.random_data_augmentation(tmp, c_lvl, c_speed)
            tmp = tmp[: self.seg_len]
            sources.append(tmp)

        mix = np.sum(np.vstack(sources), 0)

        if self.task in ["sep_noisy", "enh_single", "enh_both"]:
            sources = sources[:-1]  # discard noise

        # check for clipping
        absmax = np.max(np.abs(mix))
        if absmax > 1:
            mix = mix / absmax
            sources = [x / absmax for x in sources]

        sources = np.vstack(sources)

        # Convert to torch tensor
        mixture = torch.from_numpy(mix).float()
        # Convert sources to tensor
        sources = torch.from_numpy(sources).float()

        # Write mixture for check
        #mix_id = ids[0] + '_' + ids[1] + '_perturb'
        #dir_path ='/server/speech_data/Mandarin/aishell'
        #subdir = '0316check'
        #print("write mix file: {}".format(mix_id))
        #self.write_soundfile(mix_id, mixture, dir_path, subdir, 16000)
        #time.sleep(2)
        if not self.return_id:
            return mixture, sources

        return mixture, sources, ids
