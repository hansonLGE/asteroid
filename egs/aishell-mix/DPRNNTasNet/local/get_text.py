import argparse
import glob
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--aishelldir", required=True, type=str)
parser.add_argument("--outfile", required=True, type=str)
parser.add_argument("--split", type=str, default="train")

args = parser.parse_args()

filename2transcript = {}
with open(os.path.join(args.aishelldir, "../transcript/aishell_transcript_v0.8.txt"), "r",) as f:
    lines = f.readlines()
    for line in lines:
        key = line.split()[0]
        value = " ".join(line.split()[1:])
        filename2transcript[key] = value

aishelldir = os.path.join(args.aishelldir, args.split)
sound_paths = glob.glob(os.path.join(aishelldir, "**/*.wav"), recursive=True)
row_list = []

# Go through the sound file list
for sound_path in tqdm(sound_paths, total=len(sound_paths)):
    # Get the transcript
    filename = os.path.basename(sound_path).split('.')[0]
    if filename not in filename2transcript:
        continue
    transcript = filename2transcript[filename]

    dict1 = {}
    dict1["utt_id"] = filename
    dict1["text"] = transcript
    row_list.append(dict1)

df = pd.DataFrame(row_list)
df.to_csv(args.outfile, index=False)
