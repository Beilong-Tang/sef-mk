#!/bin/bash
export CUDA_VISIBLE_DEVICES=2


#download conformer wav2lm kmeans


audiodir=testaudio/ori
ls ${audiodir}/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > testaudio.scp

out=testaudio/anon
conformer=ckpt/librispeech_conformer_e_50.pth


flag=user # ='user' or 'attacker'
km_dir=ckpt/$flag/
ls ${km_dir}/*.pt | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > kmeans_$flag.scp


python inference_audio_multi_kmeans.py \
    --audio_scp testaudio.scp \
    --output_dir $out \
    --kmeans_scp kmeans_$flag.scp \
    --ckpt $conformer \
