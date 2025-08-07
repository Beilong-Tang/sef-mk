#!/bin/bash

# This script extracts the WavLM for LibriSpeech
#

set -e
set -u
set -o pipefail


#######
# DDP #
#######

num_proc=8 # How many processes to run in parallel 
gpus="cuda:0 cuda:1 cuda:6 cuda:7" # gpus to be used

# LibriSpeech data
###########
# out dir #
###########
out_dir=/DKUdata/tangbl/data/wavlm/librispeech/layer6 # output directory

########
# Data #
########
scp_list=("/DKUdata/tangbl/data/librispeech/test-clean/clean.scp" \
          "/DKUdata/tangbl/data/librispeech/clean.scp") # A list of scp of audio data
type=("test" "train") # The name corresponding to each item in the scp_list. (Output folder name)

#########
# Model #
#########
ckpt="/DKUdata/tangbl/wavlm/WavLM-Large.pt" # Your WavLM-large ckpt path

# Iterate using indices
for ((i=0; i<${#scp_list[@]}; i++)); do
    type=${type[$i]}
    echo "Processing $type"
    scp_file=${scp_list[$i]}
    python extract_wavlm.py --scp_file $scp_file \
      --ckpt $ckpt --output $out_dir/$type \
      --num_proc $num_proc --gpus $gpus
done

echo "Everything Done..."
