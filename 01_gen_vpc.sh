#!/bin/bash
export CUDA_VISIBLE_DEVICES=2


#download conformer wav2lm kmeans


out=anon_speech_mk
conformer=ckpt/librispeech_conformer_e_50.pth

flag=attacker
km_attacker_dir=ckpt/$flag/
ls ${km_attacker_dir}/*.pt | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > kmeans_$flag.scp

for dset in  libri_dev_enrolls libri_test_enrolls train-clean-360; do


python inference_audio_multi_kmeans.py \
    --audio_scp data/$dset/wav.scp \
    --output_dir $out/$dset \
    --kmeans_scp kmeans_$flag.scp \
    --ckpt $conformer \

done


flag=user
km_user_dir=ckpt/$flag/

ls ${km_user_dir}/*.pt | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > kmeans_$flag.scp

for dset in  libri_dev_{trials_f,trials_m} \
		libri_test_{trials_f,trials_m} \
		IEMOCAP_dev IEMOCAP_test; do

python inference_audio_multi_kmeans.py \
    --audio_scp data/$dset/wav.scp \
    --output_dir $out/$dset \
    --kmeans_scp kmeans_$flag.scp \
    --ckpt $conformer \

done



